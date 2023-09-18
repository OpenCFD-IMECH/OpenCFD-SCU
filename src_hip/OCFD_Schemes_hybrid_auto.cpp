#include "hip/hip_runtime.h"
#include <math.h>
#include "parameters.h"
#include "cuda_commen.h"
#include "commen_kernel.h"

#include "parameters_d.h"
#include "OCFD_warp_shuffle.h"
#include "cuda_utility.h"
#include "OCFD_Schemes_hybrid_auto.h"
#include "OCFD_Schemes_Choose.h"
#include "OCFD_bound_Scheme.h"
#include "OCFD_Schemes.h"
#include "OCFD_mpi_dev.h"
#include "OCFD_mpi.h"
#include "OCFD_IO_mpi.h"

#ifdef __cplusplus 
extern "C"{
#endif

void Set_Scheme_HybridAuto(hipStream_t *stream){
    Comput_P(pd_d, pT_d, pP_d, stream);

    if(HybridAuto.Style == 1){

        Comput_grad(pP_d, stream);

        modify_NT(stream);

        if(HybridAuto.IF_Smooth_dp == 1) Smoothing_dp(stream);

        Patch_zones(stream);

        Boundary_dp(stream);

        Comput_Scheme_point(stream);

    }else if(HybridAuto.Style == 2){

        Comput_Scheme_point_Jameson(stream);

    }

}


//---------------------------------------------------Comput_P--------------------------------------------------------
__global__ void Comput_P_kernel(cudaField d, cudaField T, cudaField P, REAL p00, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(P, x, y, z) = p00 * get_Field_LAP(d, x, y, z) * get_Field_LAP(T, x, y, z);
    }
}

void Comput_P(cudaField *d, cudaField *T, cudaField *P, hipStream_t *stream){
    cudaJobPackage job(dim3(0, 0, 0) , dim3(nx+2*LAP, ny+2*LAP, nz+2*LAP));

    dim3 size, griddim, blockdim;
	jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    REAL p00 = 1.0/(Gamma*Ama*Ama);
    
    CUDA_LAUNCH((hipLaunchKernelGGL(Comput_P_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *d, *T, *P, p00, job)));
}
//--------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------Comput_grad_P----------------------------------------------------
__device__ REAL warpReduce(REAL mySum){
    mySum += __shfl_xor_double(mySum, 32, hipWarpSize);
    mySum += __shfl_xor_double(mySum, 16, hipWarpSize);
    mySum += __shfl_xor_double(mySum,  8, hipWarpSize);
    mySum += __shfl_xor_double(mySum,  4, hipWarpSize);
    mySum += __shfl_xor_double(mySum,  2, hipWarpSize);
    mySum += __shfl_xor_double(mySum,  1, hipWarpSize);
    return mySum;
}

__global__ void Comput_grad1_kernel(cudaField pk, cudaField pi, cudaField ps, cudaField Akx, cudaField Aky, 
    cudaField Akz, cudaField Aix, cudaField Aiy, cudaField Aiz, cudaField Asx, cudaField Asy, cudaField Asz, 
    int SMEMDIM, cudaField grad_f, REAL *g_odata, cudaJobPackage job){
    HIP_DYNAMIC_SHARED( REAL, shared)
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int Id = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int warpId  = Id / hipWarpSize;
    unsigned int laneIdx = Id % hipWarpSize;
    REAL grad_f0 = 0.;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL px = get_Field(pk, x, y, z) * get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP) 
                + get_Field(pi, x, y, z) * get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP)
                + get_Field(ps, x, y, z) * get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);

        REAL py = get_Field(pk, x, y, z) * get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP) 
                + get_Field(pi, x, y, z) * get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP)
                + get_Field(ps, x, y, z) * get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);

        REAL pz = get_Field(pk, x, y, z) * get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP) 
                + get_Field(pi, x, y, z) * get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP)
                + get_Field(ps, x, y, z) * get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);

        get_Field(grad_f, x, y, z) = grad_f0 = sqrt(px*px + py*py + pz*pz);
    }

    grad_f0 = warpReduce(grad_f0);

    if(laneIdx == 0) shared[warpId] = grad_f0;
    __syncthreads();

    grad_f0 = (Id < SMEMDIM)?shared[Id]:0;

    if(warpId == 0) grad_f0 = warpReduce(grad_f0);
    if(Id == 0) g_odata[blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z] = grad_f0;
}

__global__ void add_kernel(REAL *g_odata, int g_odata_size){
    HIP_DYNAMIC_SHARED( REAL, shared)
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int warpId  = threadIdx.x / hipWarpSize;
    unsigned int laneIdx = threadIdx.x % hipWarpSize;

    REAL grad_f0 = 0.;
    if(x < g_odata_size) grad_f0 = g_odata[x];

    grad_f0 = warpReduce(grad_f0);
    if(laneIdx == 0) shared[warpId] = grad_f0;
    __syncthreads();

    grad_f0 = (threadIdx.x < 8)?shared[laneIdx]:0;

    if(warpId == 0) grad_f0 = warpReduce(grad_f0);

    if(x >= gridDim.x) g_odata[x] = 0.0;

    if(threadIdx.x == 0) g_odata[blockIdx.x] = grad_f0;
}

__global__ void Comput_grad2_kernel(cudaField grad_f, REAL grad_f_av1, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field(grad_f, x, y, z) = get_Field(grad_f, x, y, z)*grad_f_av1;
    }
}

void Comput_grad(cudaField *P, hipStream_t *stream){
    cudaField Pk_d, Pi_d, Ps_d;

    Pk_d.pitch = pdu_d->pitch; Pk_d.ptr = pdu_d->ptr;
    Pi_d.pitch = pdu_d->pitch; Pi_d.ptr = pdu_d->ptr + pdu_d->pitch*ny*nz;
    Ps_d.pitch = pdu_d->pitch; Ps_d.ptr = pdu_d->ptr + 2 * pdu_d->pitch*ny*nz;
    grad_P.pitch = pdu_d->pitch; grad_P.ptr = pdu_d->ptr + 3 * pdu_d->pitch*ny*nz;

    cudaJobPackage job(dim3(LAP, LAP, LAP) , dim3(nx+LAP, ny+LAP, nz+LAP));

    OCFD_dx0(*P, Pk_d, job, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
    OCFD_dy0(*P, Pi_d, job, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
    OCFD_dz0(*P, Ps_d, job, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);


    dim3 size, griddim, blockdim;
    job.setup(dim3(0, 0, 0), dim3(nx, ny, nz));
	jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    REAL *g_odata;
    REAL *Sum = (REAL *)malloc(sizeof(REAL));

    unsigned int g_odata_size = griddim.x*griddim.y*griddim.z;
    CUDA_LAUNCH(( hipMalloc((REAL **)&g_odata, g_odata_size*sizeof(REAL)) ));

    int SMEMDIM = blockdim.x*blockdim.y*blockdim.z/64;   //Warpsize is 64
    CUDA_LAUNCH((hipLaunchKernelGGL(Comput_grad1_kernel, dim3(griddim), dim3(blockdim), SMEMDIM, *stream, Pk_d, Pi_d, Ps_d, *pAkx_d, *pAky_d, 
        *pAkz_d, *pAix_d, *pAiy_d, *pAiz_d, *pAsx_d, *pAsy_d, *pAsz_d, SMEMDIM, grad_P, g_odata, job)));

    dim3 blockdim_sum(512);
    dim3 griddim_sum(g_odata_size); 

    do{
        griddim_sum.x = (griddim_sum.x + blockdim_sum.x - 1)/blockdim_sum.x;
        CUDA_LAUNCH(( hipLaunchKernelGGL(add_kernel, dim3(griddim_sum), dim3(blockdim_sum), 8, *stream, g_odata, g_odata_size) ));
    } while(griddim_sum.x > 1);

    CUDA_LAUNCH(( hipMemcpy(Sum, g_odata, sizeof(REAL), hipMemcpyDeviceToHost) ));
    CUDA_LAUNCH(( hipFree(g_odata) ));

    REAL grad_f_av, grad_f_av1;

    MPI_Allreduce(Sum, &grad_f_av, 1, OCFD_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD);

    grad_f_av = grad_f_av/(NX_GLOBAL * NY_GLOBAL * NZ_GLOBAL);
    grad_f_av1 = 1.0/grad_f_av;

    CUDA_LAUNCH((hipLaunchKernelGGL(Comput_grad2_kernel, dim3(griddim), dim3(blockdim), 0, *stream, grad_P, grad_f_av1, job)));
}
//----------------------------------------------------------------------------------------------------------

//---------------------------------------------Modify Negative T--------------------------------------------
__global__ void ana_NT_kernel(cudaField T, cudaField grad_f, REAL P_intvs, cudaJobPackage job){
    // field with LAP
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL t;
        t = get_Field_LAP(T , x,y,z);
        if(t < 0){
            t = get_Field_LAP(T , x-1 , y , z) + get_Field_LAP(T , x+1 , y , z)
               +get_Field_LAP(T , x , y-1 , z) + get_Field_LAP(T , x , y+1 , z)
               +get_Field_LAP(T , x , y , z-1) + get_Field_LAP(T , x , y , z+1);
            get_Field_LAP(T , x,y,z) = t/6.0;

            get_Field(grad_f, x, y, z) = fmax(10.*get_Field_LAP(grad_f, x, y, z), P_intvs + 1.);
        }
	}
}


void modify_NT(hipStream_t *stream){
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , nz );
    cudaJobPackage job(dim3(LAP,LAP,LAP) , dim3(nx_lap,ny_lap,nz_lap));
    hipLaunchKernelGGL(ana_NT_kernel, dim3(griddim ), dim3(blockdim), 0, *stream, *pT_d, grad_P,  HybridAuto.P_intvs[1], job);
}



//----------------------------------------------Smoothing_dp------------------------------------------------
__global__ void Modify_P_kernel(cudaField f, cudaField grad_f, REAL P_intvs, cudaJobPackage job){
    unsigned int x = (blockDim.x * blockIdx.x + threadIdx.x) + job.start.x;
	unsigned int y = (blockDim.y * blockIdx.y + threadIdx.y) + job.start.y;
    unsigned int z = (blockDim.z * blockIdx.z + threadIdx.z) + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL ff;
        get_Field_LAP(f, x, y, z) = ff = get_Field(grad_f, x-LAP, y-LAP, z-LAP);
        if(ff >= P_intvs) get_Field_LAP(f, x, y, z) = 3*ff;
    }
}

__global__ void Modify_grad_inner_kernel(cudaField f, cudaField grad_f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field(grad_f, x-LAP, y-LAP, z-LAP) = get_Field_LAP(f, x, y, z)/3.0 
                            + (get_Field_LAP(f, x+1, y, z) + get_Field_LAP(f, x-1, y, z)
                            +  get_Field_LAP(f, x, y+1, z) + get_Field_LAP(f, x, y-1, z)
                            +  get_Field_LAP(f, x, y, z+1) + get_Field_LAP(f, x, y, z-1))/9.0;
    }
}

__global__ void Modify_grad_outer_kernel(cudaField f, cudaField grad_f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field(grad_f, x, y, z) = get_Field_LAP(f, x+LAP, y+LAP, z+LAP); 
    }
}

void Smoothing_dp(hipStream_t *stream){
    REAL P_intvs = HybridAuto.P_intvs[HybridA_Stage - 1];

    cudaJobPackage job(dim3(LAP, LAP, LAP) , dim3(nx+LAP, ny+LAP, nz+LAP));
    dim3 size, griddim, blockdim;

    jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);
    CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_P_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, P_intvs, job) ));

    exchange_boundary_xyz_Async_packed_dev(pP, pPP_d, stream);

    CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_inner_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job) ))

	//---------------------------------------------------------------------------------------------
	if (npx == 0 && Iperiodic[0] == 0)
	{	
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(0, 0, 0), dim3(1, ny, nz));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, 1, ny, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job_outer) ))
	}
	if (npx == NPX0 - 1 && Iperiodic[0] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(nx-1, 0, 0), dim3(nx, ny, nz));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, 1, ny, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job_outer) ))
	}
	//---------------------------------------------------------------------------------------------
	if (npy == 0 && Iperiodic[1] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(0, 0, 0), dim3(nx, 1, nz));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, 1, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job_outer) ))
	}

	if (npy == NPY0 - 1 && Iperiodic[1] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(0, ny-1, 0), dim3(nx, ny, nz));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, 1, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job_outer) ))
	}
	//----------------------------------------------------------------------------------------------
	if (npz == 0 && Iperiodic[2] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(0, 0, 0), dim3(nx, ny, 1));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, 1);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job_outer) ))
	}

	if (npz == NPZ0 - 1 && Iperiodic[2] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(0, 0, nz-1), dim3(nx, ny, nz));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, 1);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_grad_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job_outer) ))
	}
}
//-------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------Patch_zones----------------------------------------------------
__global__ void Patch_zones_kernel(cudaField grad_f, REAL Pa_zones, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field(grad_f, x, y, z) = Pa_zones;
    }
}

void Patch_zones(hipStream_t *stream){
    int node_ib, node_ie, node_jb, node_je, node_kb, node_ke;
    int ib, ie, jb, je, kb, ke;

    for(int i = 0; i < HybridAuto.Num_Patch_zones; i++){
        int (*HybridAuto_zones)[6] = (int(*)[6])HybridAuto.zones;
        int flag_i = 1, flag_j = 1, flag_k = 1;

        get_i_node(HybridAuto_zones[i][0], &node_ib, &ib);
        get_i_node(HybridAuto_zones[i][1], &node_ie, &ie);
        get_j_node(HybridAuto_zones[i][2], &node_jb, &jb);
        get_j_node(HybridAuto_zones[i][3], &node_je, &je);
        get_k_node(HybridAuto_zones[i][4], &node_kb, &kb);
        get_k_node(HybridAuto_zones[i][5], &node_ke, &ke);

        if(node_ib < npx) ib = 0;
        if(node_ib > npx) flag_i = 0;
        if(node_ie > npx) ie = nx;
        if(node_ie < npx) flag_i = 0;

        if(node_jb < npy) jb = 0;
        if(node_jb > npy) flag_j = 0;
        if(node_je > npy) je = ny;
        if(node_je < npy) flag_j = 0;

        if(node_kb < npz) kb = 0;
        if(node_kb > npz) flag_k = 0;
        if(node_ke > npz) ke = nz;
        if(node_ke < npz) flag_k = 0;

        if(flag_i*flag_j*flag_k != 0){
            cudaJobPackage job(dim3(ib, jb, kb) , dim3(ie, je, ke));

            dim3 size, griddim, blockdim;
            jobsize(&job, &size);
            cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

            REAL Pa_zones = HybridAuto.Pa_zones[i];
            CUDA_LAUNCH(( hipLaunchKernelGGL(Patch_zones_kernel, dim3(griddim), dim3(blockdim), 0, *stream, grad_P, Pa_zones, job) ));
        }
    }
}
//---------------------------------------------------------------------------------------------------------------


//----------------------------------------------Boundary_dp-----------------------------------------------------
__global__ void Modify_P_all_kernel(cudaField f, cudaField grad_f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + f.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        get_Field_LAP(f, x, y, z, offset) = get_Field(grad_f, x, y, z);
    }
}

__global__ void Modify_x_P_outer_kernel(cudaField f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(f, x-1, y, z) = get_Field_LAP(f, x, y, z); 
    }
}

__global__ void Modify_x_M_outer_kernel(cudaField f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(f, x+1, y, z) = get_Field_LAP(f, x, y, z); 
    }
}

__global__ void Modify_y_P_outer_kernel(cudaField f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(f, x, y-1, z) = get_Field_LAP(f, x, y, z); 
    }
}

__global__ void Modify_y_M_outer_kernel(cudaField f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(f, x, y+1, z) = get_Field_LAP(f, x, y, z); 
    }
}

__global__ void Modify_z_P_outer_kernel(cudaField f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(f, x, y, z-1) = get_Field_LAP(f, x, y, z); 
    }
}

__global__ void Modify_z_M_outer_kernel(cudaField f, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(f, x, y, z+1) = get_Field_LAP(f, x, y, z); 
    }
}

void Boundary_dp(hipStream_t *stream){
    cudaJobPackage job(dim3(LAP, LAP, LAP) , dim3(nx+LAP, ny+LAP, nz+LAP));
    dim3 size, griddim, blockdim;

    jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);
    CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_P_all_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, grad_P, job) ));

    exchange_boundary_xyz_Async_packed_dev(pP, pPP_d, stream);

    //---------------------------------------------------------------------------------------------
	if (npx == 0 && Iperiodic[0] == 0)
	{	
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(LAP, LAP, LAP), dim3(LAP+1, ny+LAP, nz+LAP));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, 1, ny, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_x_P_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, job_outer) ))
	}
	if (npx == NPX0 - 1 && Iperiodic[0] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(nx+LAP-1, LAP, LAP), dim3(nx+LAP, ny+LAP, nz+LAP));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, 1, ny, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_x_M_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, job_outer) ))
	}
	//---------------------------------------------------------------------------------------------
	if (npy == 0 && Iperiodic[1] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(LAP, LAP, LAP), dim3(nx+LAP, LAP+1, nz+LAP));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, 1, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_y_P_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, job_outer) ))
	}

	if (npy == NPY0 - 1 && Iperiodic[1] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(LAP, ny+LAP-1, LAP), dim3(nx+LAP, ny+LAP, nz+LAP));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, 1, nz);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_y_M_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, job_outer) ))
	}
	//----------------------------------------------------------------------------------------------
	if (npz == 0 && Iperiodic[2] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(LAP, LAP, LAP), dim3(nx+LAP, ny+LAP, LAP+1));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, 1);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_z_P_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, job_outer) ))
	}

	if (npz == NPZ0 - 1 && Iperiodic[2] == 0)
	{
		dim3 griddim, blockdim;
		cudaJobPackage job_outer(dim3(LAP, LAP, nz+LAP-1), dim3(nx+LAP, ny+LAP, nz+LAP));

		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, 1);
        CUDA_LAUNCH(( hipLaunchKernelGGL(Modify_z_M_outer_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, job_outer) ))
	}
}
//---------------------------------------------------------------------------------------------------------------------


//-----------------------------------------------Comput_Scheme_point---------------------------------------------------
__global__ void Comput_Scheme_point_x_kernel(cudaField P, cudaField_int scheme, REAL P_intvs1, REAL P_intvs2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + P.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        REAL dp0 = 0.5 * (get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x+1, y, z, offset));
        int kp = 1;
        if(dp0 > P_intvs1) kp += 1;
        if(dp0 > P_intvs2) kp += 1;
        *(scheme.ptr + (x + scheme.pitch *(y + (z)*ny_d))) = kp;
    }
}

__global__ void Comput_Scheme_point_y_kernel(cudaField P, cudaField_int scheme, REAL P_intvs1, REAL P_intvs2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + P.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        REAL dp0 = 0.5 * (get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y+1, z, offset));
        int kp = 1;
        if(dp0 > P_intvs1) kp += 1;
        if(dp0 > P_intvs2) kp += 1;
        *(scheme.ptr + (x + scheme.pitch *(y + (z)*(ny_d + 1)))) = kp;
    }
}

__global__ void Comput_Scheme_point_z_kernel(cudaField P, cudaField_int scheme, REAL P_intvs1, REAL P_intvs2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + P.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        REAL dp0 = 0.5 * (get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y, z+1, offset));
        int kp = 1;
        if(dp0 > P_intvs1) kp += 1;
        if(dp0 > P_intvs2) kp += 1;
        *(scheme.ptr + (x + scheme.pitch *(y + (z)*ny_d))) = kp;
    }
}

void Comput_Scheme_point(hipStream_t *stream){

    dim3 size, griddim, blockdim;
    cudaJobPackage job(dim3(LAP-1, LAP, LAP), dim3(nx+LAP, ny+LAP, nz+LAP));

	jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);
    
    CUDA_LAUNCH(( hipLaunchKernelGGL(Comput_Scheme_point_x_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, *HybridAuto.scheme_x, HybridAuto.P_intvs[0], HybridAuto.P_intvs[1], job) ));//HybridA_Stage == 3

    job.setup(dim3(LAP, LAP-1, LAP), dim3(nx+LAP, ny+LAP, nz+LAP));

    jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    CUDA_LAUNCH(( hipLaunchKernelGGL(Comput_Scheme_point_y_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, *HybridAuto.scheme_y, HybridAuto.P_intvs[0], HybridAuto.P_intvs[1], job) ));//HybridA_Stage == 3

    job.setup(dim3(LAP, LAP, LAP-1), dim3(nx+LAP, ny+LAP, nz+LAP));

    jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    CUDA_LAUNCH(( hipLaunchKernelGGL(Comput_Scheme_point_z_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pPP_d, *HybridAuto.scheme_z, HybridAuto.P_intvs[0], HybridAuto.P_intvs[1], job) ));//HybridA_Stage == 3
}


__global__ void Comput_Scheme_point_x_Jameson_kernel(cudaField P, cudaField_int scheme, REAL P_intvs1, REAL P_intvs2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + P.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        REAL dp0 = fabs(-get_Field_LAP(P, x-1, y, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x+1, y, z, offset))/
        (get_Field_LAP(P, x-1, y, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x+1, y, z, offset));

        dp0 += fabs(-get_Field_LAP(P, x, y-1, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x, y+1, z, offset))/
        (get_Field_LAP(P, x, y-1, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y+1, z, offset));

        dp0 += fabs(-get_Field_LAP(P, x, y, z-1, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x, y, z+1, offset))/
        (get_Field_LAP(P, x, y, z-1, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y, z+1, offset));

        int kp = 1;
        if(dp0 > P_intvs1) kp += 1;
        if(dp0 > P_intvs2) kp += 1;
        *(scheme.ptr + (x + scheme.pitch *(y + (z)*ny_d))) = kp;
    }
}

__global__ void Comput_Scheme_point_y_Jameson_kernel(cudaField P, cudaField_int scheme, REAL P_intvs1, REAL P_intvs2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + P.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        REAL dp0 = fabs(-get_Field_LAP(P, x-1, y, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x+1, y, z, offset))/
        (get_Field_LAP(P, x-1, y, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x+1, y, z, offset));

        dp0 += fabs(-get_Field_LAP(P, x, y-1, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x, y+1, z, offset))/
        (get_Field_LAP(P, x, y-1, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y+1, z, offset));

        dp0 += fabs(-get_Field_LAP(P, x, y, z-1, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x, y, z+1, offset))/
        (get_Field_LAP(P, x, y, z-1, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y, z+1, offset));

        int kp = 1;
        if(dp0 > P_intvs1) kp += 1;
        if(dp0 > P_intvs2) kp += 1;
        *(scheme.ptr + (x + scheme.pitch *(y + (z)*(ny_d + 1)))) = kp;
    }
}

__global__ void Comput_Scheme_point_z_Jameson_kernel(cudaField P, cudaField_int scheme, REAL P_intvs1, REAL P_intvs2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int offset = job.start.x + P.pitch*(job.start.y + ny_2lap_d*job.start.z);
    
    if(x < (job.end.x - job.start.x) && y < (job.end.y - job.start.y) && z < (job.end.z - job.start.z)){
        REAL dp0 = fabs(-get_Field_LAP(P, x-1, y, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x+1, y, z, offset))/
        (get_Field_LAP(P, x-1, y, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x+1, y, z, offset));

        dp0 += fabs(-get_Field_LAP(P, x, y-1, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x, y+1, z, offset))/
        (get_Field_LAP(P, x, y-1, z, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y+1, z, offset));

        dp0 += fabs(-get_Field_LAP(P, x, y, z-1, offset) + 2*get_Field_LAP(P, x, y, z, offset) - get_Field_LAP(P, x, y, z+1, offset))/
        (get_Field_LAP(P, x, y, z-1, offset) + 2*get_Field_LAP(P, x, y, z, offset) + get_Field_LAP(P, x, y, z+1, offset));

        int kp = 1;
        if(dp0 > P_intvs1) kp += 1;
        if(dp0 > P_intvs2) kp += 1;
        *(scheme.ptr + (x + scheme.pitch *(y + (z)*ny_d))) = kp;
    }
}

void Comput_Scheme_point_Jameson(hipStream_t *stream){

    dim3 size, griddim, blockdim;
    cudaJobPackage job(dim3(LAP-1, LAP, LAP), dim3(nx+LAP, ny+LAP, nz+LAP));

	jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);
    
    CUDA_LAUNCH(( hipLaunchKernelGGL(Comput_Scheme_point_x_Jameson_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pP_d, *HybridAuto.scheme_x, HybridAuto.P_intvs[0], HybridAuto.P_intvs[1], job) ));//HybridA_Stage == 3

    job.setup(dim3(LAP, LAP-1, LAP), dim3(nx+LAP, ny+LAP, nz+LAP));

    jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    CUDA_LAUNCH(( hipLaunchKernelGGL(Comput_Scheme_point_y_Jameson_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pP_d, *HybridAuto.scheme_y, HybridAuto.P_intvs[0], HybridAuto.P_intvs[1], job) ));//HybridA_Stage == 3

    job.setup(dim3(LAP, LAP, LAP-1), dim3(nx+LAP, ny+LAP, nz+LAP));

    jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    CUDA_LAUNCH(( hipLaunchKernelGGL(Comput_Scheme_point_z_Jameson_kernel, dim3(griddim), dim3(blockdim), 0, *stream, *pP_d, *HybridAuto.scheme_z, HybridAuto.P_intvs[0], HybridAuto.P_intvs[1], job) ));//HybridA_Stage == 3
}


void HybridAuto_scheme_IO(){
    memcpy_All_int(scheme_x, HybridAuto.scheme_x->ptr, HybridAuto.scheme_x->pitch, D2H, nx+1, ny, nz);
    memcpy_All_int(scheme_y, HybridAuto.scheme_y->ptr, HybridAuto.scheme_y->pitch, D2H, nx, ny+1, nz);
    memcpy_All_int(scheme_z, HybridAuto.scheme_z->ptr, HybridAuto.scheme_z->pitch, D2H, nx, ny, nz+1);
    memcpy_All(pP, pPP_d->ptr , pPP_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);

    FILE *fp; 
    char fp_name[120];

    if(my_id == 0){
        sprintf(fp_name, "Scheme_x%08d.dat", Istep);
        fp = fopen(fp_name, "w");
    }
    write_2d_XY(fp, NZ_GLOBAL/2, nx+1, ny, 0, scheme_x, pP);

    if(my_id == 0){
        fclose(fp);
        sprintf(fp_name, "Scheme_y%08d.dat", Istep);
        fp = fopen(fp_name, "w");
    }
    write_2d_XY(fp, NZ_GLOBAL/2, nx, ny+1, 0, scheme_y, pP);
    
    if(my_id == 0){
        fclose(fp);
        sprintf(fp_name, "Scheme_z%08d.dat", Istep);
        fp = fopen(fp_name, "w");
    }
    write_2d_XY(fp, NZ_GLOBAL/2, nx, ny, 0, scheme_z, pP);
    
    if(my_id == 0) fclose(fp);
}


void HybridAuto_scheme_Proportion(){
    memcpy_All_int(scheme_x, HybridAuto.scheme_x->ptr, HybridAuto.scheme_x->pitch, D2H, nx+1, ny, nz);
    memcpy_All_int(scheme_y, HybridAuto.scheme_y->ptr, HybridAuto.scheme_y->pitch, D2H, nx, ny+1, nz);
    memcpy_All_int(scheme_z, HybridAuto.scheme_z->ptr, HybridAuto.scheme_z->pitch, D2H, nx, ny, nz+1);

    double type1 = 0.0, type2 = 0.0, type3 = 0.0;
    double Sum_type1, Sum_type2, Sum_type3;

    int tmp = (nx + 1) * ny * nz;
    for(int i = 0; i < tmp; i++){
        if(*(scheme_x + i) == 1){ 
            type1 += 1.0;
        }else if(*(scheme_x + i) == 2){
            type2 += 1.0;
        }else{
            type3 += 1.0;
        }
    }

    type1 /= NY_GLOBAL*NZ_GLOBAL;
    type2 /= NY_GLOBAL*NZ_GLOBAL;
    type3 /= NY_GLOBAL*NZ_GLOBAL;

    MPI_Reduce(&type1, &Sum_type1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&type2, &Sum_type2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&type3, &Sum_type3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    tmp = NX_GLOBAL + NPX0;

    char scheme_percent[] = "The first type scheme of Hybrid schemes in direction %s is \033[34m%lf%\033[0m, second is"
 "\033[34m%lf%\033[0m， third is \033[34m%lf%\033[0m\n";

    if(my_id == 0) printf(scheme_percent, "X", Sum_type1/tmp, Sum_type2/tmp, Sum_type3/tmp);

    type1 = 0.0; type2 = 0.0; type3 = 0.0;

    tmp = nx * (ny + 1) * nz;
    for(int i = 0; i < tmp; i++){
        if(*(scheme_y + i) == 1){ 
            type1 += 1.0;
        }else if(*(scheme_y + i) == 2){
            type2 += 1.0;
        }else{
            type3 += 1.0;
        }
    }

    type1 /= NX_GLOBAL*NZ_GLOBAL;
    type2 /= NX_GLOBAL*NZ_GLOBAL;
    type3 /= NX_GLOBAL*NZ_GLOBAL;

    MPI_Reduce(&type1, &Sum_type1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&type2, &Sum_type2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&type3, &Sum_type3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    tmp = (NY_GLOBAL + NPY0);

    if(my_id == 0) printf(scheme_percent, "Y", Sum_type1/tmp, Sum_type2/tmp, Sum_type3/tmp);

    type1 = 0.0; type2 = 0.0; type3 = 0.0;

    tmp = nx * ny * (nz + 1);
    for(int i = 0; i < tmp; i++){
        if(*(scheme_z + i) == 1){ 
            type1 += 1.0;
        }else if(*(scheme_z + i) == 2){
            type2 += 1.0;
        }else{
            type3 += 1.0;
        }
    }

    type1 /= NX_GLOBAL*NY_GLOBAL;
    type2 /= NX_GLOBAL*NY_GLOBAL;
    type3 /= NX_GLOBAL*NY_GLOBAL;

    MPI_Reduce(&type1, &Sum_type1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&type2, &Sum_type2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&type3, &Sum_type3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    tmp = (NZ_GLOBAL + NPZ0);

    if(my_id == 0) printf(scheme_percent, "Z", Sum_type1/tmp, Sum_type2/tmp, Sum_type3/tmp);

}


__device__ int get_Hyscheme_flag_p_kernel(int flagxyz, dim3 coords, cudaField_int scheme, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;
    int Hyscheme_flag;

	switch(flagxyz){
		case 1:
		case 4:
        Hyscheme_flag = *(scheme.ptr + (x + 1 - LAP + scheme.pitch *(y - LAP + (z - LAP)*ny_d)));
		return Hyscheme_flag;
		break;

	    case 2:
	    case 5:
        Hyscheme_flag = *(scheme.ptr + (x - LAP + scheme.pitch *(y + 1 - LAP + (z - LAP)*(ny_d + 1))));
	    return Hyscheme_flag;
	    break;

		case 3:
		case 6:
        Hyscheme_flag = *(scheme.ptr + (x - LAP + scheme.pitch *(y - LAP + (z + 1 - LAP)*ny_d)));
		return Hyscheme_flag;
		break;
	}

    return 0;
}


__global__ void OCFD_HybridAuto_P_kernel(dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job){
    HIP_DYNAMIC_SHARED( REAL, sort)
	dim3 coords;
	REAL stencil[8];
    int Hyscheme_flag;
	int ia1 = -3; int ib1 = 4;
	

    for(int i = 0; i < 5; i++){

		int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

		if(flag != 0){
			REAL tmp_r, tmp_l;

            if(i == 0) Hyscheme_flag = get_Hyscheme_flag_p_kernel(flagxyzb.x, coords, scheme, job);

            flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

			if(flag != 0){
                if(Hyscheme_flag == 1){
                    tmp_r = OCFD_OMP6_kernel_P(0, &stencil[0]);
                }else if(Hyscheme_flag == 2){
                    tmp_r = OCFD_weno7_kernel_P(&stencil[0]);
                }else{
                    tmp_r = OCFD_NND2_kernel_P(&stencil[2]);
                }
            }

			tmp_l = __shfl_up_double(tmp_r, 1, hipWarpSize);

            if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
		}
	}

}


__global__ void OCFD_HybridAuto_P_Jameson_kernel(dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job){
    HIP_DYNAMIC_SHARED( REAL, sort)
	dim3 coords;
	REAL stencil[8];
    int Hyscheme_flag;
	int ia1 = -3; int ib1 = 4;
	

    for(int i = 0; i < 5; i++){

		int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

		if(flag != 0){
			REAL tmp_r, tmp_l;

            if(i == 0) Hyscheme_flag = get_Hyscheme_flag_p_kernel(flagxyzb.x, coords, scheme, job);

            flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

			if(flag != 0){
                if(Hyscheme_flag == 1){
                    tmp_r = OCFD_UP7_kernel_P(&stencil[0]);
                }else if(Hyscheme_flag == 2){
                    tmp_r = OCFD_weno7_kernel_P(&stencil[0]);
                }else{
                    tmp_r = OCFD_weno5_kernel_P(&stencil[1]);
                }
            }

			tmp_l = __shfl_up_double(tmp_r, 1, hipWarpSize);

            if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
		}
	}

}


__device__ int get_Hyscheme_flag_m_kernel(int flagxyz, dim3 coords, cudaField_int scheme, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;
    int Hyscheme_flag;

	switch(flagxyz){
		case 1:
		case 4:
        Hyscheme_flag = *(scheme.ptr + (x - LAP + scheme.pitch *(y - LAP + (z - LAP)*ny_d)));
		return Hyscheme_flag;
        break;

	    case 2:
	    case 5:
        Hyscheme_flag = *(scheme.ptr + (x - LAP + scheme.pitch *(y - LAP + (z - LAP)*(ny_d + 1))));
	    return Hyscheme_flag;
	    break;

		case 3:
		case 6:
        Hyscheme_flag = *(scheme.ptr + (x - LAP + scheme.pitch *(y - LAP + (z - LAP)*ny_d)));
		return Hyscheme_flag;
		break;
	}

    return 0;
}


__global__ void OCFD_HybridAuto_M_kernel(dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job){
    HIP_DYNAMIC_SHARED( REAL, sort)
	dim3 coords;
	REAL stencil[8];
    int Hyscheme_flag;

	int ia1 = -4; int ib1 = 3;


    for(int i = 0; i < 5; i++){

        int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);
    
        if(flag != 0){
            REAL tmp_r, tmp_l; 

            if(i == 0) Hyscheme_flag = get_Hyscheme_flag_m_kernel(flagxyzb.x, coords, scheme, job);
    
            flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  
    
            if(flag != 0){
                if(Hyscheme_flag == 1){
                    tmp_r = OCFD_OMP6_kernel_M(0, &stencil[0]);
                }else if(Hyscheme_flag == 2){
                    tmp_r = OCFD_weno7_kernel_M(&stencil[1]);
                }else{
                    tmp_r = OCFD_NND2_kernel_M(&stencil[3]);
                }
            }
    
            tmp_l = __shfl_up_double(tmp_r, 1, hipWarpSize);
    
            if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
        }
    }
}


__global__ void OCFD_HybridAuto_M_Jameson_kernel(dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job){
    HIP_DYNAMIC_SHARED( REAL, sort)
	dim3 coords;
	REAL stencil[8];
    int Hyscheme_flag;

	int ia1 = -4; int ib1 = 3;


    for(int i = 0; i < 5; i++){

        int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);
    
        if(flag != 0){
            REAL tmp_r, tmp_l; 

            if(i == 0) Hyscheme_flag = get_Hyscheme_flag_m_kernel(flagxyzb.x, coords, scheme, job);
    
            flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  
    
            if(flag != 0){
                if(Hyscheme_flag == 1){
                    tmp_r = OCFD_UP7_kernel_M(&stencil[0]);
                }else if(Hyscheme_flag == 2){
                    tmp_r = OCFD_weno7_kernel_M(&stencil[1]);
                }else{
                    tmp_r = OCFD_weno5_kernel_M(&stencil[2]);
                }
            }
    
            tmp_l = __shfl_up_double(tmp_r, 1, hipWarpSize);
    
            if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
        }
    }
}



#ifdef __cplusplus
}
#endif

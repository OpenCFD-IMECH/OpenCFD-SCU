#include "hip/hip_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "parameters.h"
#include "parameters_d.h"
#include "utility.h"
#include "io_warp.h"
#include "cuda_commen.h"
#include "cuda_utility.h"

#include "OCFD_boundary_init.h"
#include "OCFD_boundary_compression_conner.h"

#ifdef __cplusplus
extern "C"{
#endif

extern int x_begin;
extern cudaField *pub1_d;
extern cudaField *pfx_d;
extern cudaField *pgz_d;
extern REAL *fait;
extern REAL *TM;


__global__ void do_ub1_inlet_kernel(cudaField d, cudaField u, cudaField v, cudaField w, cudaField T, cudaField ub1, cudaJobPackage job){
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if(y < job.end.y && z < job.end.z){
        unsigned int ylap = y - LAP;
        for(int i = 0; i <= LAP; i++){
            get_Field_LAP(d, i, y, z) = *(ub1.ptr + ylap);
            get_Field_LAP(u, i, y, z) = *(ub1.ptr + ylap + ub1.pitch * 1);
            get_Field_LAP(v, i, y, z) = *(ub1.ptr + ylap + ub1.pitch * 2);
            get_Field_LAP(w, i, y, z) = 0.;
            get_Field_LAP(T, i, y, z) = *(ub1.ptr + ylap + ub1.pitch * 3);
        }
    }
}


__global__ void do_ub1_top_kernel(cudaField d, cudaField u, cudaField v, cudaField w, cudaField T, cudaField ub1, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if(x < job.end.x && z < job.end.z){
        get_Field_LAP(d, x, ny_lap_d - 1, z) = *(ub1.ptr + ny_d - 1);
        get_Field_LAP(u, x, ny_lap_d - 1, z) = *(ub1.ptr + ub1.pitch * 1 + ny_d - 1);
        get_Field_LAP(v, x, ny_lap_d - 1, z) = *(ub1.ptr + ub1.pitch * 2 + ny_d - 1);
        get_Field_LAP(w, x, ny_lap_d - 1, z) = 0.;
        get_Field_LAP(T, x, ny_lap_d - 1, z) = *(ub1.ptr + ub1.pitch * 3 + ny_d - 1);

    }
}


__global__ void do_wall_dist_kernel(cudaField d, cudaField u, cudaField v, cudaField w, cudaField T, 
REAL HT, REAL epsl, cudaField fx, cudaField gz, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if(x < job.end.x && z < job.end.z){
        unsigned int xlap = x - LAP;
        unsigned int zlap = z - LAP;
 
        get_Field_LAP(u, x, LAP, z) = 0.;
        get_Field_LAP(v, x, LAP, z) = epsl * (*(fx.ptr + xlap)) * (*(gz.ptr + zlap)) * HT;
        get_Field_LAP(w, x, LAP, z) = 0.;

    }
}


__global__ void do_wall_Tp1_kernel(cudaField d, cudaField T, cudaField u, cudaField v, cudaField w, REAL tw, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    REAL pw;

    if(x < job.end.x && z < job.end.z){
        pw = (18. * get_Field_LAP(d, x, LAP + 1, z) * get_Field_LAP(T, x, LAP + 1, z)
             - 9. * get_Field_LAP(d, x, LAP + 2, z) * get_Field_LAP(T, x, LAP + 2, z)
             + 2. * get_Field_LAP(d, x, LAP + 3, z) * get_Field_LAP(T, x, LAP + 3, z))/ 11.;


        get_Field_LAP(T, x, LAP, z) = tw;
        get_Field_LAP(d, x, LAP, z) = pw / get_Field_LAP(T, x, LAP, z);


        for(int i = 0; i < LAP; i++){
            get_Field_LAP(d, x, i, z) =  get_Field_LAP(d, x, 2*LAP-i, z);
            get_Field_LAP(u, x, i, z) = -get_Field_LAP(u, x, 2*LAP-i, z);
            get_Field_LAP(v, x, i, z) = -get_Field_LAP(v, x, 2*LAP-i, z);
            get_Field_LAP(w, x, i, z) = -get_Field_LAP(w, x, 2*LAP-i, z);
            get_Field_LAP(T, x, i, z) =  get_Field_LAP(T, x, 2*LAP-i, z);
        }
        
    }
}


__global__ void do_wall_Tp2_kernel(cudaField d, cudaField T, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    REAL pw;

    if(x < job.end.x && z < job.end.z){

        get_Field_LAP(T, x, LAP, z) = (18. * get_Field_LAP(T, x, LAP + 1, z)
                                       -9. * get_Field_LAP(T, x, LAP + 2, z)
                                       +2. * get_Field_LAP(T, x, LAP + 3, z)) / 11.;
        pw = (18. * get_Field_LAP(d, x, LAP + 1, z) * get_Field_LAP(T, x, LAP + 1, z)
             - 9. * get_Field_LAP(d, x, LAP + 2, z) * get_Field_LAP(T, x, LAP + 2, z)
             + 2. * get_Field_LAP(d, x, LAP + 3, z) * get_Field_LAP(T, x, LAP + 3, z)) / 11.;
        get_Field_LAP(d, x, LAP, z) = pw / get_Field_LAP(T, x, LAP, z);
        
    }
}


void bc_user_Compression_conner(){
    
//---------------------boundary condition at i=1 -------------------------------//
    if(npx == 0){
        dim3 blockdim , griddim;
        cal_grid_block_dim(&griddim, &blockdim, 1, BlockDimY, BlockDimZ, 1, ny, nz);
        cudaJobPackage job( dim3(LAP, LAP ,LAP) , dim3(LAP+1, ny_lap, nz_lap) );
        CUDA_LAUNCH(( hipLaunchKernelGGL(do_ub1_inlet_kernel, dim3(griddim), dim3(blockdim), 0, 0, *pd_d, *pu_d, *pv_d, *pw_d, *pT_d, *pub1_d, job) ));
   
    }

    if(npy == NPY0 - 1){
        int x_do = nx - x_begin;
        dim3 blockdim , griddim;
        cal_grid_block_dim(&griddim, &blockdim, BlockDimX, 1, BlockDimZ, x_do, 1, nz);
        cudaJobPackage job( dim3(x_begin + LAP, LAP, LAP) , dim3(nx_lap, LAP+1, nz_lap) );
        //cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, LAP+1, nz_lap) );
        CUDA_LAUNCH(( hipLaunchKernelGGL(do_ub1_top_kernel, dim3(griddim), dim3(blockdim), 0, 0, *pd_d, *pu_d, *pv_d, *pw_d, *pT_d, *pub1_d, job) ));
    }
    
    REAL ht = 0.;

    if(BETA > 0.){
        for(int m = 0; m < MTMAX; m++){
            //ht = ht + TM[m] * sin((m + 1)*BETA*tt + 2.*PI*fait[m]);
            ht = ht + TM[m] * sin((m + 1)*BETA*tt);
        }
    }else{
        ht = 1.;
    }


//---------------------wall-boundary-condition-at-j=1---------------------------//
    if(npy == 0){
        dim3 blockdim , griddim;
        cal_grid_block_dim(&griddim, &blockdim, BlockDimX, 1, BlockDimZ, nx, 1, nz);
        cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, LAP+1, nz_lap) );
        CUDA_LAUNCH(( hipLaunchKernelGGL(do_wall_dist_kernel, dim3(griddim), dim3(blockdim), 0, 0, *pd_d, *pu_d, *pv_d, *pw_d, *pT_d, ht, EPSL, *pfx_d, *pgz_d, job) ));

    
//Comput pressure and temperature-correction caused by non-wall-normal mesh ----//
        if(IFLAG_WALL_NOT_NORMAL == 0){
            if(TW > 0){
            dim3 blockdim , griddim;
                cal_grid_block_dim(&griddim, &blockdim, BlockDimX, 1, BlockDimZ, nx, 1, nz);
                cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, LAP+1, nz_lap) );
                CUDA_LAUNCH(( hipLaunchKernelGGL(do_wall_Tp1_kernel, dim3(griddim), dim3(blockdim), 0, 0, *pd_d, *pT_d, *pu_d, *pv_d, *pw_d, TW, job) ));
            }else{
            dim3 blockdim , griddim;
                cal_grid_block_dim(&griddim, &blockdim, BlockDimX, 1, BlockDimZ, nx, 1, nz);
                cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, LAP+1, nz_lap) );
                CUDA_LAUNCH(( hipLaunchKernelGGL(do_wall_Tp2_kernel, dim3(griddim), dim3(blockdim), 0, 0, *pd_d, *pT_d, job) ));
            }
        }else{
            printf("Now non-normal wall is not supported\n");
        }
    }
}


//void get_ht_multifrequancy(REAL HT, REAL TT, int MT_MAX, REAL beta){
//    HT = 0.;
//
//    if(beta > 0.){
//        for(int m = 0; m < MT_MAX; m++){
//            HT = HT + TM[m] * sin((m + 1)*beta*TT + 2.*PI*fait[m]);
//        }
//    }else{
//        HT = 1.;
//    }
//
//}
#ifdef __cplusplus
}
#endif

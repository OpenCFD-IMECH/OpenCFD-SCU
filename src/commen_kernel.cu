#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "parameters.h"
#include "parameters_d.h"

#ifdef __cplusplus
extern "C"{
#endif
__global__ void cuda_mem_value_init(REAL value, REAL *ptr, unsigned int pitch, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < size_x && y < size_y && z < size_z)
    {
        *(ptr + x + pitch * (y + z * size_y)) = value;
    }
}
void cuda_mem_value_init_warp(REAL value, REAL *ptr, unsigned int pitch, unsigned int size_x, unsigned int size_y, unsigned int size_z){
    dim3 griddim ;
    dim3 blockdim ;

    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , size_x , size_y , size_z);
    cuda_mem_value_init<<<griddim , blockdim>>>(value, ptr , pitch , size_x , size_y , size_z);
}


/* ========================= */
__global__ void pri_to_cons_kernel(cudaSoA pcons , cudaField pd , cudaField pu , cudaField pv , cudaField pw , cudaField pT ,cudaJobPackage job ){
    // eyes on cells WITHOUT LAPs
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + job.start.x; 
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + job.start.y; 
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z + job.start.z;

    REAL d,u,v,w,T;
    if(x<job.end.x && y<job.end.y && z<job.end.z){
        { 
            // d
            d = get_Field_LAP(pd , x+LAP,y+LAP,z+LAP);
            get_SoA(pcons , x,y,z , 0) = d;
        }
        {
            // u*d
            u = get_Field_LAP(pu , x+LAP , y+LAP , z+LAP);
            get_SoA(pcons , x,y,z , 1) = u*d;
        }
        {
            // v*d
            v = get_Field_LAP(pv , x+LAP , y+LAP , z+LAP);
            get_SoA(pcons , x,y,z , 2) = v*d;
        }
        {
            // w*d
            w = get_Field_LAP(pw , x+LAP , y+LAP , z+LAP);
            get_SoA(pcons , x,y,z , 3) = w*d ;
        }
        {
            // E
            T = get_Field_LAP(pT , x+LAP , y+LAP , z+LAP);
            get_SoA(pcons , x,y,z , 4) = d*( (u*u + v*v + w*w)*0.5 + Cv_d * T);
        }
    }
}
void pri_to_cons_kernel_warp(cudaSoA *pcons , cudaField *pd , cudaField *pu , cudaField *pv , cudaField *pw , cudaField *pT , cudaJobPackage job_in , dim3 blockdim_in ){
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , blockdim_in.x , blockdim_in.y , blockdim_in.z , job_in.end.x , job_in.end.y , job_in.end.z);
    cudaJobPackage job;
    CUDA_LAUNCH(( pri_to_cons_kernel<<<griddim , blockdim>>>(*pcons , *pd , *pu , *pv , *pw , *pT , job_in) ))
}


/* ========================= */

__global__ void cons_to_pri_kernel(cudaSoA f, cudaSoA spec, cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField P , cudaJobPackage job){
    // eyes on no-lap region
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL d1,u1,v1,w1,T1,d2,T2;
        
        d1 = get_SoA(f, x, y, z, 0);
        get_Field_LAP(d, x+LAP, y+LAP, z+LAP) = d2 = d1;

        u1 = get_SoA(f, x, y, z, 1);
        u1 = u1/d1;
        get_Field_LAP(u, x+LAP, y+LAP, z+LAP) = u1;

        v1 = get_SoA(f, x, y, z, 2);
        v1 = v1/d1;
        get_Field_LAP(v, x+LAP, y+LAP, z+LAP) = v1;

        w1 = get_SoA(f, x, y, z, 3);
        w1 = w1/d1;
        get_Field_LAP(w, x+LAP, y+LAP, z+LAP) = w1;

        T1 = get_SoA(f, x, y, z, 4);
        get_Field_LAP(T, x+LAP, y+LAP, z+LAP) = T2 = (T1 - 0.5*d1*(u1*u1 + v1*v1 + w1*w1))/(d1*Cv_d);
        // T1 = T1/(d1*Cv_d);
        //T1 = T1 - 0.5*(u1*u1 + v1*v1 + w1*w1)/d1;
        // get_Field_LAP(P, x+LAP, y+LAP, z+LAP) = T1*(Gamma_d - 1.0);
        // get_Field_LAP(T , x+LAP , y+LAP , z+LAP) = T1/Cv_d;
        get_Field_LAP(P, x+LAP, y+LAP, z+LAP) = T2*d2/(Gamma_d*Ama_d*Ama_d);

        REAL sum = 0.0;

        for (int n=0; n<NSPECS; ++n) {
            sum += get_SoA_LAP(spec, x+LAP, y+LAP, z+LAP, n);
        }
        for (int n=0; n<NSPECS; ++n) {
            get_SoA_LAP(spec, x+LAP, y+LAP, z+LAP, n) *= d2/sum;
        }
    }
}

__global__ void get_spec_kernel(cudaSoA f, cudaField O , cudaField O2 , cudaField N , cudaField NO , cudaField N2 , cudaJobPackage job){
    // eyes on no-lap region
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL s1, s2, s3, s4, s5;
        
        s1 = get_SoA_LAP(f, x+LAP, y+LAP, z+LAP, 0);
        get_Field_LAP(O, x+LAP, y+LAP, z+LAP) = s1;

        s2 = get_SoA_LAP(f, x+LAP, y+LAP, z+LAP, 1);
        get_Field_LAP(O2, x+LAP, y+LAP, z+LAP) = s2;

        s3 = get_SoA_LAP(f, x+LAP, y+LAP, z+LAP, 2);
        get_Field_LAP(N, x+LAP, y+LAP, z+LAP) = s3;

        s4 = get_SoA_LAP(f, x+LAP, y+LAP, z+LAP, 3);
        get_Field_LAP(NO, x+LAP, y+LAP, z+LAP) = s4;

        s5 = get_SoA_LAP(f, x+LAP, y+LAP, z+LAP, 4);
        get_Field_LAP(N2, x+LAP, y+LAP, z+LAP) = s5;
    }
}

void get_duvwT()
{
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx,ny,nz);
    cudaJobPackage job(dim3(0,0,0) , dim3(nx,ny,nz));

    CUDA_LAUNCH(( cons_to_pri_kernel<<<griddim , blockdim>>>(*pf_d , *pspec_d, *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pP_d , job) ))
}

void get_spec()
{
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx,ny,nz);
    cudaJobPackage job(dim3(0,0,0) , dim3(nx,ny,nz));

    CUDA_LAUNCH(( get_spec_kernel<<<griddim , blockdim>>>(*pspec_d , *pO_d , *pO2_d , *pN_d , *pNO_d , *pN2_d , job) ))
}

// -----Computation of viscousity ---------------------------------------------

__global__ void get_Amu_kernal(cudaField Amu , cudaField T , cudaJobPackage job){
    // eyes on field without LAP
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL tmp = get_Field_LAP(T , x+LAP , y+LAP , z+LAP);
        get_Field(Amu , x,y,z) = amu_C0_d * sqrt(tmp * tmp * tmp) / (Tsb_d + tmp);
    }
}
void get_Amu()
{
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx,ny,nz);
    cudaJobPackage job(dim3(0,0,0) , dim3(nx,ny,nz));

    CUDA_LAUNCH(( get_Amu_kernal<<<griddim , blockdim>>>(*pAmu_d , *pT_d , job) ))
}

/* ======================================================== */

__global__ void sound_speed_kernel(cudaField T , cudaField cc , cudaJobPackage job){
    // eyes on no-lap region
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if( x<job.end.x && y<job.end.y && z<job.end.z){
        get_Field_LAP(cc , x,y,z) = sqrt( get_Field_LAP(T , x,y,z) )/Ama_d;
    }
}

/* ============================================================================== */

// out += xf
__global__ void YF_Pe_XF(cudaField yF , cudaField xF , cudaField AJac , cudaJobPackage job){
    // WITHOUT LAPs
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if( x<job.end.x && y<job.end.y && z<job.end.z){
	    REAL ajac;
	    ajac = get_Field_LAP(AJac, x+LAP, y+LAP, z+LAP);
        atomicAdd(yF.ptr + (x + yF.pitch*(y + ny_d*z)), ajac * get_Field(xF, x, y, z));
    }
}

// out = xf+yf
__global__ void ZF_e_XF_P_YF(cudaField out , cudaField xF , cudaField yF , cudaJobPackage job){
    // WITHOUT LAPs
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if( x<job.end.x && y<job.end.y && z<job.end.z){
        get_Field(out , x,y,z) = get_Field(xF , x,y,z) + get_Field(yF , x,y,z) ;
    }
}
__global__ void ZF_e_XF_P_YF_LAP(cudaField out , cudaField xF , cudaField yF , cudaJobPackage job){
    // WITH LAPs
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

    if( x<job.end.x && y<job.end.y && z<job.end.z){
        get_Field_LAP(out , x,y,z) = get_Field_LAP(xF , x,y,z) + get_Field_LAP(yF , x,y,z) ;
    }
}

// zf += xf+yf
__global__ void ZF_Pe_XF_P_YF(cudaField zF , cudaField xF , cudaField yF , cudaField AJac , cudaJobPackage job){
    // WITHOUT LAPs
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    if( x<job.end.x && y<job.end.y && z<job.end.z){
	    REAL ajac;
	    ajac = get_Field_LAP(AJac , x+LAP , y+LAP , z+LAP);
        get_Field(zF , x,y,z) += - ajac * ( get_Field(xF , x,y,z) + get_Field(yF , x,y,z) ) ;
    }
}


//__device__ void ZF_Pe_XF_P_YF_Device(cudaField zF, cudaField xF, cudaField yF, cudaField AJac){
//    // WITHOUT LAPs
//	REAL ajac;
//	ajac = get_Field_LAP(AJac , x+LAP , y+LAP , z+LAP);
//        get_Field(zF , x,y,z) += - ajac * ( get_Field(xF , x,y,z) + get_Field(yF , x,y,z) ) ;
//}
#ifdef __cplusplus
}
#endif

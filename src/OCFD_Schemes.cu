#include <math.h>
#include "parameters.h"
#include "utility.h"
#include "OCFD_Schemes.h"
#include "OCFD_bound_Scheme.h"

#include "parameters_d.h"
#include "OCFD_warp_shuffle.h"
#include "cuda_utility.h"

#ifdef __cplusplus 
extern "C"{
#endif

__device__ int get_data0_kernel(int flagxyz, dim3 *coords, cudaField pf, REAL *stencil, int ka1, int kb1, cudaJobPackage job){
	int offset = job.start.x + pf.pitch*(job.start.y + ny_2lap_d*job.start.z);

	switch(flagxyz){
		case 1:
		case 4:
		{
			unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
			unsigned int y = coords->y = blockDim.y * blockIdx.y + threadIdx.y;
			unsigned int z = coords->z = blockDim.z * blockIdx.z + threadIdx.z;


        	if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)){
				for(int i = ka1; i <= kb1; i++){
					stencil[i-ka1] = get_Field_LAP(pf, x+i, y, z, offset);
				}
				return 1;
			}
		}
		break;

		case 2:
		case 5:
        {
            unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int y = coords->y = blockDim.y * blockIdx.y + threadIdx.y;
            unsigned int z = coords->z = blockDim.z * blockIdx.z + threadIdx.z;

            if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)){
                for(int i = ka1; i <= kb1; i++){
                    stencil[i-ka1] = get_Field_LAP(pf, x, y+i, z, offset);
                }
                return 2;
            }
        }
		break;

		case 3:
		case 6:
        {
            unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int y = coords->y = blockDim.z * blockIdx.z + threadIdx.z;
            unsigned int z = coords->z = blockDim.y * blockIdx.y + threadIdx.y;

            if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)){
                for(int i = ka1; i <= kb1; i++){
                    stencil[i-ka1] = get_Field_LAP(pf, x, y, z+i, offset);
                }
                return 3;
            }
        }
		break;
	}

	return 0;
}


__device__ void put_d0_kernel(dim3 flagxyz, dim3 coords, REAL tmp, cudaField pfy, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

	switch(flagxyz.x){
		case 1:
		case 4:
		get_Field(pfy, x-LAP, y-LAP, z-LAP) = tmp/hx_d;
		break;

		case 2:
		case 5:
		get_Field(pfy, x-LAP, y-LAP, z-LAP) = tmp/hy_d;
		break;

		case 3:
		case 6:
		get_Field(pfy, x-LAP, y-LAP, z-LAP) = tmp/hz_d;
		break;
	}
}

__device__ REAL OCFD_kernel_CD6(REAL *stencil){

	REAL tmp = (45.0*(stencil[4] - stencil[2])
			    -9.0*(stencil[5] - stencil[1])
			        +(stencil[6] - stencil[0]))/60.0;

	return tmp;
}


__global__ void OCFD_CD6_kernel(dim3 flagxyzb, cudaField pf, cudaField pfy, cudaJobPackage job){
	dim3 coords;
	REAL stencil[7], tmp;

	int ia1 = -3; int ib1 = 3;


	int flag = get_data0_kernel(flagxyzb.x, &coords, pf, &stencil[0], ia1, ib1, job);

	if(flag != 0){

		flag =  OCFD_D0bound_scheme_kernel(&tmp, flagxyzb, coords, &stencil[0], ia1, job); 


		if(flag != 0) tmp = OCFD_kernel_CD6(&stencil[0]);

	    put_d0_kernel(flagxyzb, coords, tmp, pfy, job);

    }
}


__device__ REAL OCFD_kernel_CD8(REAL *stencil){

	REAL tmp = (672.0*(stencil[5] - stencil[3])
			   -168.0*(stencil[6] - stencil[2])
			    +32.0*(stencil[7] - stencil[1])
			     -3.0*(stencil[8] - stencil[0]))/840.0;

	return tmp;
}

__global__ void OCFD_CD8_kernel(dim3 flagxyzb, cudaField pf, cudaField pfy, cudaJobPackage job){
	dim3 coords;
	REAL stencil[9], tmp;

	int ia1 = -4; int ib1 = 4;


	int flag = get_data0_kernel(flagxyzb.x, &coords, pf, &stencil[0], ia1, ib1, job);

	if(flag != 0){

		flag =  OCFD_D0bound_scheme_kernel(&tmp, flagxyzb, coords, &stencil[0], ia1, job); 

		if(flag != 0) tmp = OCFD_kernel_CD8(&stencil[0]);

	    put_d0_kernel(flagxyzb, coords, tmp, pfy, job);
    
    }
}


__device__ int get_data_kernel(int flagxyz, dim3 *coords, cudaSoA f, int num, REAL *stencil, int ka1, int kb1, REAL *sort, cudaJobPackage job){
	
	int offset = job.start.x + f.pitch*(job.start.y + ny_2lap_d*job.start.z);

	switch(flagxyz){
		case 1:
		case 4:
		{
			unsigned int x = coords->x = (blockDim.x-1) * blockIdx.x + threadIdx.x;
			unsigned int y = coords->y = blockDim.y * blockIdx.y + threadIdx.y;
			unsigned int z = coords->z = blockDim.z * blockIdx.z + threadIdx.z;

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)){
				for(int i = ka1; i <= kb1; i++){
					stencil[i-ka1] = get_SoA_LAP(f, x+i, y, z, num, offset);
				}
				return 1;
			}
		}
		break;

		case 2:
		{   
            unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int y = coords->y = (blockDim.y-1) * blockIdx.y + threadIdx.y;
            unsigned int z = coords->z = blockDim.z * blockIdx.z + threadIdx.z;

            unsigned int ID1 = 128*threadIdx.z + 16*threadIdx.x + threadIdx.y;
            unsigned int ID2 = 128*threadIdx.z + 16*threadIdx.y + threadIdx.x;

            sort[ID1] = get_SoA_LAP(f, x, y-LAP+1, z, num, offset);

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y-1) && z < (job.end.z-job.start.z))
            sort[ID1+8] = get_SoA_LAP(f, x, y+LAP+1, z, num, offset);

            __syncthreads();

            for(int i = ka1; i <= kb1; i++){
                stencil[i-ka1] = sort[ID2+i+3];
            }
			
            x = coords->x = blockDim.x * blockIdx.x + threadIdx.y;
            y = coords->y = (blockDim.y-1) * blockIdx.y + threadIdx.x;

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)) return 2;            
		}
		break;

		case 5:
		{   
            unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int y = coords->y = (blockDim.y-1) * blockIdx.y + threadIdx.y;
            unsigned int z = coords->z = blockDim.z * blockIdx.z + threadIdx.z;

            unsigned int ID1 = 128*threadIdx.z + 16*threadIdx.x + threadIdx.y;
            unsigned int ID2 = 128*threadIdx.z + 16*threadIdx.y + threadIdx.x;

            sort[ID1] = get_SoA_LAP(f, x, y-LAP, z, num, offset);

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y-1) && z < (job.end.z-job.start.z))
            sort[ID1+8] = get_SoA_LAP(f, x, y+LAP, z, num, offset);

            __syncthreads();

            for(int i = ka1; i <= kb1; i++){
                stencil[i-ka1] = sort[ID2+i+LAP];
            }
			
            x = coords->x = blockDim.x * blockIdx.x + threadIdx.y;
            y = coords->y = (blockDim.y-1) * blockIdx.y + threadIdx.x;

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)) return 2;            
		}
		break;

		case 3:
		{
            unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int y = coords->y = blockDim.z * blockIdx.z + threadIdx.z;
            unsigned int z = coords->z = (blockDim.y-1) * blockIdx.y + threadIdx.y;

            unsigned int ID1 = 128*threadIdx.z + 16*threadIdx.x + threadIdx.y;
            unsigned int ID2 = 128*threadIdx.z + 16*threadIdx.y + threadIdx.x;

            sort[ID1] = get_SoA_LAP(f, x, y, z-LAP+1, num, offset);

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z-1))
            sort[ID1+8] = get_SoA_LAP(f, x, y, z+LAP+1, num, offset);

            __syncthreads();

            for(int i = ka1; i <= kb1; i++){
                stencil[i-ka1] = sort[ID2+i+3];
            }
			
            x = coords->x = blockDim.x * blockIdx.x + threadIdx.y;
            z = coords->z = (blockDim.y-1) * blockIdx.y + threadIdx.x;

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)) return 3;     
		}
		break;

		case 6:
		{
            unsigned int x = coords->x = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int y = coords->y = blockDim.z * blockIdx.z + threadIdx.z;
            unsigned int z = coords->z = (blockDim.y-1) * blockIdx.y + threadIdx.y;

            unsigned int ID1 = 128*threadIdx.z + 16*threadIdx.x + threadIdx.y;
            unsigned int ID2 = 128*threadIdx.z + 16*threadIdx.y + threadIdx.x;

            sort[ID1] = get_SoA_LAP(f, x, y, z-LAP, num, offset);

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z-1))
            sort[ID1+8] = get_SoA_LAP(f, x, y, z+LAP, num, offset);

            __syncthreads();

            for(int i = ka1; i <= kb1; i++){
                stencil[i-ka1] = sort[ID2+i+LAP];
            }
			
            x = coords->x = blockDim.x * blockIdx.x + threadIdx.y;
            z = coords->z = (blockDim.y-1) * blockIdx.y + threadIdx.x;

			if(x < (job.end.x-job.start.x) && y < (job.end.y-job.start.y) && z < (job.end.z-job.start.z)) return 3;            
		}
		break;
	}

	return 0;
}


__device__ void put_du_p_kernel(dim3 flagxyz, dim3 coords, REAL tmp_r, REAL tmp_l, cudaSoA du, int num, cudaField Ajac, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

	switch(flagxyz.x){
		case 1:
		case 4:
		if(flagxyz.x == 1 && flagxyz.z == 1 && coords.x == 1){
			//get_SoA(du, x-LAP, y-LAP, z-LAP, num) += 0;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
			//get_SoA(du, x-LAP, y-LAP, z-LAP, num) += -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hx_d;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hx_d);
		}
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP) = (tmp_r - tmp_l)/hx_d;
		break;

		case 2:
		case 5:
		if(flagxyz.x == 2 && flagxyz.z == 1 && coords.y == 1){
			//get_SoA(du, x-LAP, y-LAP, z-LAP, num) += 0;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
			//get_SoA(du, x-LAP, y-LAP, z-LAP, num) += -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hy_d;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hy_d);
            
		}
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP) = (tmp_r - tmp_l)/hy_d;
		break;

		case 3:
		case 6:
		if(flagxyz.x == 3 && flagxyz.z == 1 && coords.z == 1){
			//get_SoA(du, x-LAP, y-LAP, z-LAP, num) += 0;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
			//get_SoA(du, x-LAP, y-LAP, z-LAP, num) += -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hz_d;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hz_d);
		}
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP) = (tmp_r - tmp_l)/hz_d;
		break;
	}
}


__device__ void put_du_m_kernel(dim3 flagxyz, dim3 coords, REAL tmp_r, REAL tmp_l, cudaSoA du, int num, cudaField Ajac, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

	switch(flagxyz.x){
		case 1:
		case 4:
		if(flagxyz.x == 4 && flagxyz.z == 1 && coords.x == job.end.x-job.start.x-1){
			//get_SoA(du, x-LAP-1, y-LAP, z-LAP, num) += 0;
            atomicAdd(du.ptr + (x - LAP - 1 + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
			//get_SoA(du, x-LAP-1, y-LAP, z-LAP, num) += -get_Field_LAP(Ajac, x-1, y, z)*(tmp_r - tmp_l)/hx_d;
            atomicAdd(du.ptr + (x - LAP - 1 + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x-1, y, z)*(tmp_r - tmp_l)/hx_d);
		}
		//get_Field(Ajac, x-LAP-1, y-LAP, z-LAP) = (tmp_r - tmp_l)/hx_d;
		break;

		case 2:
		case 5:
		if(flagxyz.x == 5 && flagxyz.z == 1 && coords.y == job.end.y-job.start.y-1){
			//get_SoA(du, x-LAP, y-LAP-1, z-LAP, num) += 0;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP - 1 + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
			//get_SoA(du, x-LAP, y-LAP-1, z-LAP, num) += -get_Field_LAP(Ajac, x, y-1, z)*(tmp_r - tmp_l)/hy_d;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP - 1 + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y-1, z)*(tmp_r - tmp_l)/hy_d);
		}
		//get_Field(Ajac, x-LAP, y-LAP-1, z-LAP) = (tmp_r - tmp_l)/hy_d;
		break;

		case 3:
		case 6:
		if(flagxyz.x == 6 && flagxyz.z == 1 && coords.z == job.end.z-job.start.z-1){
			//get_SoA(du, x-LAP, y-LAP, z-LAP-1, num) += 0;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP - 1 + (num)*nz_d))), 0);
		}else{
			//get_SoA(du, x-LAP, y-LAP, z-LAP-1, num) += -get_Field_LAP(Ajac, x, y, z-1)*(tmp_r - tmp_l)/hz_d;
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP - 1 + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z-1)*(tmp_r - tmp_l)/hz_d);
		}
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP-1) = (tmp_r - tmp_l)/hz_d;
		break;
	}
}

// =================================================================================================================================== //



//---------------------------------------------------------WENO7_SYMBO_P-------------------------------------------------------------//
//   7th order WENO-SYMBO scheme (Bandwith-Optimized Symmetric WENO scheme), see in Martin et al, J. Comput. Phys. 220, 270-289
// -----The difference between WENO-SYMOO and WENO-SYMBO is that the coefficients of  is different, other is the same.

__device__ REAL OCFD_weno7_SYMBO_kernel_P(int WENO_LMT_FLAG, REAL *stencil){

	REAL S0, S1, S2, S3, S4;
	REAL tmp, tmp1, TVmin = 0, TVmax = 1;
			
	if(WENO_LMT_FLAG == 1){
		S0 = fabs(stencil[1] - stencil[0]) + fabs(stencil[2] - stencil[1]) + fabs(stencil[3] - stencil[2]);
		S1 = fabs(stencil[2] - stencil[1]) + fabs(stencil[3] - stencil[2]) + fabs(stencil[4] - stencil[3]);
		S2 = fabs(stencil[3] - stencil[2]) + fabs(stencil[4] - stencil[3]) + fabs(stencil[5] - stencil[4]);
		S3 = fabs(stencil[4] - stencil[3]) + fabs(stencil[5] - stencil[4]) + fabs(stencil[6] - stencil[5]);
		S4 = fabs(stencil[5] - stencil[4]) + fabs(stencil[6] - stencil[5]) + fabs(stencil[7] - stencil[6]);

		tmp   = fmin(S0,S1);
		tmp1  = fmin(S2,S3);
		tmp   = fmin(tmp,tmp1);
		TVmin = fmin(tmp ,S4);

		tmp   = fmax(S0,S1);
		tmp1  = fmax(S2,S3);
		tmp   = fmax(tmp,tmp1);
		TVmax = fmax(tmp ,S4);
	}
		
	if(TVmax < WENO_TV_Limiter_d*TVmin && TVmax < WENO_TV_MAX_d){
		S0 = 0.0401954833730;
		S1 = 0.2493800006710;
		S2 = 0.4802686256260;
		S3 = 0.2009775476730;
		S4 = 0.0291783426580;
	}else{

		S0 = 0.0; S1 = 0.0; S2 = 0.0; S3 = 0.0; S4 =0.0; 

		// 1st  
		tmp =  -2.0*stencil[0] +  9.0*stencil[1] - 18.0*stencil[2] + 11.0*stencil[3]; S0 += 720.0*tmp*tmp;
		tmp =       stencil[1] -  6.0*stencil[2] +  3.0*stencil[3] +  2.0*stencil[4]; S1 += 720.0*tmp*tmp;
		tmp =  -2.0*stencil[2] -  3.0*stencil[3] +  6.0*stencil[4] -      stencil[5]; S2 += 720.0*tmp*tmp;
		tmp = -11.0*stencil[3] + 18.0*stencil[4] -  9.0*stencil[5] +  2.0*stencil[6]; S3 += 720.0*tmp*tmp;
		tmp = -26.0*stencil[4] + 57.0*stencil[5] - 42.0*stencil[6] + 11.0*stencil[7]; S4 += 720.0*tmp*tmp;

		// 2nd 
		tmp = -6.0*stencil[0] + 24.0*stencil[1] - 30.0*stencil[2] + 12.0*stencil[3]; S0 += 780.0*tmp*tmp;
		tmp =                    6.0*stencil[2] - 12.0*stencil[3] +  6.0*stencil[4]; S1 += 780.0*tmp*tmp;
		tmp =                    6.0*stencil[3] - 12.0*stencil[4] +  6.0*stencil[5]; S2 += 780.0*tmp*tmp;
		tmp = 12.0*stencil[3] - 30.0*stencil[4] + 24.0*stencil[5] -  6.0*stencil[6]; S3 += 780.0*tmp*tmp;
		tmp = 18.0*stencil[4] - 48.0*stencil[5] + 42.0*stencil[6] - 12.0*stencil[7]; S4 += 780.0*tmp*tmp;

		// 3rd 
		tmp = -6.0*stencil[0] + 18.0*( stencil[1] - stencil[2] ) + 6.0*stencil[3]; S0 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[1] + 18.0*( stencil[2] - stencil[3] ) + 6.0*stencil[4]; S1 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[2] + 18.0*( stencil[3] - stencil[4] ) + 6.0*stencil[5]; S2 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[3] + 18.0*( stencil[4] - stencil[5] ) + 6.0*stencil[6]; S3 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[4] + 18.0*( stencil[5] - stencil[6] ) + 6.0*stencil[7]; S4 += 781.0*tmp*tmp;

		{
			tmp   = fmax(S0,S1);
			tmp1  = fmax(S2,S3);
			tmp   = fmax(tmp,tmp1);
			S4 = fmax(tmp ,S4);
		}
		
		{
			REAL tmp2, tmp3;
		    tmp =  (0.0401954833730)*(2.592e-6+S1)*(2.592e-6+S2)*(2.592e-6+S3)*(2.592e-6+S4);
		    tmp1 = (0.2493800006710)*(2.592e-6+S0)*(2.592e-6+S2)*(2.592e-6+S3)*(2.592e-6+S4);
		    tmp2 = (0.4802686256260)*(2.592e-6+S0)*(2.592e-6+S1)*(2.592e-6+S3)*(2.592e-6+S4);
		    tmp3 = (0.2009775476730)*(2.592e-6+S0)*(2.592e-6+S1)*(2.592e-6+S2)*(2.592e-6+S4);
		    S4 =   (0.0291783426580)*(2.592e-6+S0)*(2.592e-6+S1)*(2.592e-6+S2)*(2.592e-6+S3);
		    S3 = tmp3;
		    S2 = tmp2;
		    S1 = tmp1;
			S0 = tmp;
		}
	}
    
	REAL am=S0+S1+S2+S3+S4;
    
	tmp1 = 0.0;
	tmp = -3.0*stencil[0] + 13.0*stencil[1] - 23.0*stencil[2] + 25.0*stencil[3]; tmp1 += S0*tmp;
	tmp =      stencil[1] -  5.0*stencil[2] + 13.0*stencil[3] +  3.0*stencil[4]; tmp1 += S1*tmp;
	tmp =     -stencil[2] +  7.0*stencil[3] +  7.0*stencil[4] -      stencil[5]; tmp1 += S2*tmp;
	tmp =  3.0*stencil[3] + 13.0*stencil[4] -  5.0*stencil[5] +      stencil[6]; tmp1 += S3*tmp;
	tmp = 25.0*stencil[4] - 23.0*stencil[5] + 13.0*stencil[6] -  3.0*stencil[7]; tmp1 += S4*tmp;

	tmp1 /= (12.0*am);

	return tmp1;
}


__global__ void OCFD_weno7_SYMBO_P_kernel(int i, int WENO_LMT_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[8];

	int ia1 = -3; int ib1 = 4;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
		REAL tmp_r, tmp_l;

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job); 

		if(flag != 0) tmp_r = OCFD_weno7_SYMBO_kernel_P(WENO_LMT_FLAG, &stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);

	}
}

//=====================================================================================================================================================//


//----------------------------------------------------------------WENO7_SYMBO_M------------------------------------------------------------------------//
__device__ REAL OCFD_weno7_SYMBO_kernel_M(int WENO_LMT_FLAG, REAL *stencil){

	REAL S0, S1, S2, S3, S4;
	REAL tmp, tmp1, TVmin = 0, TVmax = 1;

	if(WENO_LMT_FLAG == 1){
		S0 = fabs(stencil[7] - stencil[6]) + fabs(stencil[6] - stencil[5]) + fabs(stencil[5] - stencil[4]);
		S1 = fabs(stencil[6] - stencil[5]) + fabs(stencil[5] - stencil[4]) + fabs(stencil[4] - stencil[3]);
		S2 = fabs(stencil[5] - stencil[4]) + fabs(stencil[4] - stencil[3]) + fabs(stencil[3] - stencil[2]);
		S3 = fabs(stencil[4] - stencil[3]) + fabs(stencil[3] - stencil[2]) + fabs(stencil[2] - stencil[1]);
		S4 = fabs(stencil[3] - stencil[2]) + fabs(stencil[2] - stencil[1]) + fabs(stencil[1] - stencil[0]);

		tmp   = fmin(S0,S1);
		tmp1  = fmin(S2,S3);
		tmp   = fmax(tmp,tmp1);
		TVmax = fmax(tmp ,S4);
	}

	if(TVmax < WENO_TV_Limiter_d*TVmin && TVmax < WENO_TV_MAX_d){
		S0 = 0.0401954833730;
		S1 = 0.2493800006710;
		S2 = 0.4802686256260;
		S3 = 0.2009775476730;
		S4 = 0.0291783426580;
	}else{
		S0 = 0.0; S1 = 0.0; S2 = 0.0; S3 = 0.0; S4 =0.0; 

		// 1st  
		tmp =  -2.0*stencil[7] +  9.0*stencil[6] - 18.0*stencil[5] + 11.0*stencil[4]; S0 += 720.0*tmp*tmp;
		tmp =       stencil[6] -  6.0*stencil[5] +  3.0*stencil[4] +  2.0*stencil[3]; S1 += 720.0*tmp*tmp;
		tmp =  -2.0*stencil[5] -  3.0*stencil[4] +  6.0*stencil[3] -      stencil[2]; S2 += 720.0*tmp*tmp;
		tmp = -11.0*stencil[4] + 18.0*stencil[3] -  9.0*stencil[2] +  2.0*stencil[1]; S3 += 720.0*tmp*tmp;
		tmp = -26.0*stencil[3] + 57.0*stencil[2] - 42.0*stencil[1] + 11.0*stencil[0]; S4 += 720.0*tmp*tmp;

		// 2nd 
		tmp = -6.0*stencil[7] + 24.0*stencil[6] - 30.0*stencil[5] + 12.0*stencil[4]; S0 += 780.0*tmp*tmp;
		tmp =              		 6.0*stencil[5] - 12.0*stencil[4] +  6.0*stencil[3]; S1 += 780.0*tmp*tmp;
		tmp =                    6.0*stencil[4] - 12.0*stencil[3] +  6.0*stencil[2]; S2 += 780.0*tmp*tmp;
		tmp = 12.0*stencil[4] - 30.0*stencil[3] + 24.0*stencil[2] -  6.0*stencil[1]; S3 += 780.0*tmp*tmp;
		tmp = 18.0*stencil[3] - 48.0*stencil[2] + 42.0*stencil[1] - 12.0*stencil[0]; S4 += 780.0*tmp*tmp;

		// 3rd 
		tmp = -6.0*stencil[7] + 18.0*( stencil[6] - stencil[5] ) + 6.0*stencil[4]; S0 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[6] + 18.0*( stencil[5] - stencil[4] ) + 6.0*stencil[3]; S1 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[5] + 18.0*( stencil[4] - stencil[3] ) + 6.0*stencil[2]; S2 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[4] + 18.0*( stencil[3] - stencil[2] ) + 6.0*stencil[1]; S3 += 781.0*tmp*tmp; 
		tmp = -6.0*stencil[3] + 18.0*( stencil[2] - stencil[1] ) + 6.0*stencil[0]; S4 += 781.0*tmp*tmp;

		{
			tmp  = fmax(S0,S1);
			tmp1 = fmax(S2,S3);
			tmp  = fmax(tmp,tmp1);
			S4   = fmax(tmp ,S4);
		}

		{
			REAL tmp2, tmp3;
			tmp =  (0.0401954833730)*(2.592e-6+S1)*(2.592e-6+S2)*(2.592e-6+S3)*(2.592e-6+S4);
			tmp1 = (0.2493800006710)*(2.592e-6+S0)*(2.592e-6+S2)*(2.592e-6+S3)*(2.592e-6+S4);
			tmp2 = (0.4802686256260)*(2.592e-6+S0)*(2.592e-6+S1)*(2.592e-6+S3)*(2.592e-6+S4);
			tmp3 = (0.2009775476730)*(2.592e-6+S0)*(2.592e-6+S1)*(2.592e-6+S2)*(2.592e-6+S4);
			S4 =   (0.0291783426580)*(2.592e-6+S0)*(2.592e-6+S1)*(2.592e-6+S2)*(2.592e-6+S3);
			S3 = tmp3;
			S2 = tmp2;
			S1 = tmp1;
			S0 = tmp;
		}
	}

	REAL am=S0+S1+S2+S3+S4;

	tmp1 = 0.0;
	tmp = -3.0*stencil[7] + 13.0*stencil[6] - 23.0*stencil[5] + 25.0*stencil[4]; tmp1 += S0*tmp;
	tmp =      stencil[6] -  5.0*stencil[5] + 13.0*stencil[4] +  3.0*stencil[3]; tmp1 += S1*tmp;
	tmp =    - stencil[5] +  7.0*stencil[4] +  7.0*stencil[3] -      stencil[2]; tmp1 += S2*tmp;
	tmp =  3.0*stencil[4] + 13.0*stencil[3] -  5.0*stencil[2] +      stencil[1]; tmp1 += S3*tmp;
	tmp = 25.0*stencil[3] - 23.0*stencil[2] + 13.0*stencil[1] -  3.0*stencil[0]; tmp1 += S4*tmp;

	tmp1 /= (12.0*am);

	return tmp1;
}


__global__ void OCFD_weno7_SYMBO_M_kernel(int i, int WENO_LMT_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
	extern __shared__ REAL sort[];
    dim3 coords;
	REAL stencil[8];

	int ia1 = -4; int ib1 = 3;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
	    REAL tmp_r = 0.0, tmp_l = 0.0; 

		flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);

		if(flag != 0) tmp_r = OCFD_weno7_SYMBO_kernel_M(WENO_LMT_FLAG, &stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}
//=======================================================================================================================================================//

//---------------------------------------------------------------------WENO7_P---------------------------------------------------------------------------//
__device__ REAL OCFD_weno7_kernel_P(REAL *stencil){

	REAL S0 =0.0, S1 =0.0, S2 =0.0, S3 =0.0;
	REAL tmp, tmp1, tmp2;

	tmp  = -2.0*stencil[0] +  9.0*stencil[1] - 18.0*stencil[2] + 11.0*stencil[3]; S0 += 960*tmp*tmp;
	tmp1 = -6.0*stencil[0] + 24.0*stencil[1] - 30.0*stencil[2] + 12.0*stencil[3]; S0 += (1040.0)*tmp1*tmp1;
	tmp2 = -6.0*stencil[0] + 18.0*( stencil[1] - stencil[2] ) + 6.0*stencil[3];   S0 += (1043.0)*tmp2*tmp2; 
	tmp  = tmp * tmp2; S0 += 80.0*tmp;

	tmp  =      stencil[1] - 6.0*stencil[2] +  3.0*stencil[3] + 2.0*stencil[4]; S1 += 960*tmp*tmp;
	tmp1 =                   6.0*stencil[2] - 12.0*stencil[3] + 6.0*stencil[4]; S1 += (1040.0)*tmp1*tmp1;
	tmp2 = -6.0*stencil[1] + 18.0*( stencil[2] - stencil[3] ) + 6.0*stencil[4]; S1 += (1043.0)*tmp2*tmp2; 
	tmp  = tmp * tmp2; S1 += 80.0*tmp;
		
	tmp  = -2.0*stencil[2] - 3.0*stencil[3] +  6.0*stencil[4] -   stencil[5]; S2 += 960*tmp*tmp;
	tmp1 =                 6.0*stencil[3] - 12.0*stencil[4] + 6.0*stencil[5]; S2 += (1040.0)*tmp1*tmp1;
	tmp2 = -6.0*stencil[2] + 18.0*(stencil[3] - stencil[4]) + 6.0*stencil[5]; S2 += (1043.0)*tmp2*tmp2; 
	tmp  = tmp * tmp2; S2 += 80.0*tmp;
		
	tmp  = -11.0*stencil[3] + 18.0*stencil[4] -  9.0*stencil[5] + 2.0*stencil[6]; S3 += 960*tmp*tmp;
	tmp1 =  12.0*stencil[3] - 30.0*stencil[4] + 24.0*stencil[5] - 6.0*stencil[6]; S3 += (1040.0)*tmp1*tmp1;
	tmp2 =  -6.0*stencil[3] + 18.0*( stencil[4] -  stencil[5] ) + 6.0*stencil[6];  S3 += (1043.0)*tmp2*tmp2; 
	tmp  =  tmp * tmp2; S3 += 80.0*tmp;


	tmp  =        ((3.456e-4+S1)*(3.456e-4+S1))*((3.456e-4+S2)*(3.456e-4+S2))*((3.456e-4+S3)*(3.456e-4+S3));
	tmp1 = (12.0)*((3.456e-4+S0)*(3.456e-4+S0))*((3.456e-4+S2)*(3.456e-4+S2))*((3.456e-4+S3)*(3.456e-4+S3));
	tmp2 = (18.0)*((3.456e-4+S0)*(3.456e-4+S0))*((3.456e-4+S1)*(3.456e-4+S1))*((3.456e-4+S3)*(3.456e-4+S3));
	S3   =  (4.0)*((3.456e-4+S0)*(3.456e-4+S0))*((3.456e-4+S1)*(3.456e-4+S1))*((3.456e-4+S2)*(3.456e-4+S2));
	S2   = tmp2;
	S1   = tmp1;
	S0   = tmp;
		

	REAL am=S0+S1+S2+S3;

		
	tmp1 = 0.0;
	tmp = -3.0*stencil[0] + 13.0*stencil[1] - 23.0*stencil[2] + 25.0*stencil[3]; tmp1 += S0*tmp;
	tmp =      stencil[1] -  5.0*stencil[2] + 13.0*stencil[3] +  3.0*stencil[4]; tmp1 += S1*tmp;
	tmp =     -stencil[2] +  7.0*stencil[3] +  7.0*stencil[4] -      stencil[5]; tmp1 += S2*tmp;
	tmp =  3.0*stencil[3] + 13.0*stencil[4] -  5.0*stencil[5] +      stencil[6]; tmp1 += S3*tmp;

	tmp1 /= (12.0*am);

	return tmp1;
}


__global__ void OCFD_weno7_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
	extern __shared__ REAL sort[];
    dim3 coords;
	REAL stencil[7];

	int ia1 = -3; int ib1 = 3;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
		REAL tmp_r, tmp_l;

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job); 

		if(flag != 0) tmp_r = OCFD_weno7_kernel_P(&stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);

	}
}
//==============================================================================================================================================================//


//------------------------------------------------------------------------WENO7_M-------------------------------------------------------------------------------//
__device__ REAL OCFD_weno7_kernel_M(REAL *stencil){

	REAL S0 =0.0, S1 =0.0, S2 =0.0, S3 =0.0;
	REAL tmp, tmp1, tmp2;


	tmp  = -2.0*stencil[6] +  9.0*stencil[5] - 18.0*stencil[4] + 11.0*stencil[3]; S0 += 960*tmp*tmp;
	tmp1 = -6.0*stencil[6] + 24.0*stencil[5] - 30.0*stencil[4] + 12.0*stencil[3]; S0 += (1040.0)*tmp1*tmp1;
	tmp2 = -6.0*stencil[6] + 18.0*( stencil[5] -  stencil[4] ) + 6.0*stencil[3];  S0 += (1043.0)*tmp2*tmp2; 
	tmp  = tmp * tmp2; S0 += (80.0) * tmp;
		
	tmp  =      stencil[5] -  6.0* stencil[4] + 3.0*stencil[3] + 2.0*stencil[2]; S1 += 960*tmp*tmp;
	tmp1 =                  6.0*stencil[4] - 12.0* stencil[3] +  6.0*stencil[2]; S1 += (1040.0)*tmp1*tmp1;
	tmp2 = -6.0*stencil[5] + 18.0*( stencil[4] -  stencil[3]) + 6.0*stencil[2];  S1 += (1043.0)*tmp2*tmp2; 
	tmp  = tmp *tmp2; S1 += (80.0) * tmp;
		
	tmp  =    -2.0*stencil[4] - 3.0*stencil[3] + 6.0*stencil[2] - stencil[1]; S2 += 960*tmp*tmp;
	tmp1 =                 6.0*stencil[3] - 12.0*stencil[2] + 6.0*stencil[1]; S2 += (1040.0)*tmp1*tmp1;
	tmp2 = -6.0*stencil[4] + 18.0*(stencil[3] - stencil[2]) + 6.0*stencil[1]; S2 += (1043.0)*tmp2*tmp2; 
	tmp  = tmp *tmp2; S2 += (80.0) * tmp;
		
	tmp  = -11.0*stencil[3] + 18.0*stencil[2] -  9.0*stencil[1] + 2.0*stencil[0]; S3 += 960*tmp*tmp;
	tmp1 =  12.0*stencil[3] - 30.0*stencil[2] + 24.0*stencil[1] - 6.0*stencil[0]; S3 += (1040.0)*tmp1*tmp1;
	tmp2 =  -6.0*stencil[3] + 18.0*( stencil[2] -  stencil[1] ) + 6.0*stencil[0]; S3 += (1043.0)*tmp2*tmp2; 
	tmp  =  tmp *tmp2; S3 += (80.0) * tmp;



	tmp  =        ((3.456e-4+S1)*(3.456e-4+S1))*((3.456e-4+S2)*(3.456e-4+S2))*((3.456e-4+S3)*(3.456e-4+S3));
	tmp1 = (12.0)*((3.456e-4+S0)*(3.456e-4+S0))*((3.456e-4+S2)*(3.456e-4+S2))*((3.456e-4+S3)*(3.456e-4+S3));
	tmp2 = (18.0)*((3.456e-4+S0)*(3.456e-4+S0))*((3.456e-4+S1)*(3.456e-4+S1))*((3.456e-4+S3)*(3.456e-4+S3));
	S3   =  (4.0)*((3.456e-4+S0)*(3.456e-4+S0))*((3.456e-4+S1)*(3.456e-4+S1))*((3.456e-4+S2)*(3.456e-4+S2));
	S2   = tmp2;
	S1   = tmp1;
	S0   = tmp;
		

	REAL am=S0+S1+S2+S3;


	tmp1 = 0.0;
	tmp = -3.0*stencil[6] + 13.0*stencil[5] - 23.0*stencil[4] + 25.0*stencil[3]; tmp1 += S0*tmp;
	tmp =      stencil[5] -  5.0*stencil[4] + 13.0*stencil[3] +  3.0*stencil[2]; tmp1 += S1*tmp;
	tmp =     -stencil[4] +  7.0*stencil[3] +  7.0*stencil[2] -      stencil[1]; tmp1 += S2*tmp;
	tmp =  3.0*stencil[3] + 13.0*stencil[2] -  5.0*stencil[1] +      stencil[0]; tmp1 += S3*tmp;

	tmp1 /= (12.0*am);

	return tmp1;
}


__global__ void OCFD_weno7_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
	extern __shared__ REAL sort[];
    dim3 coords;
	REAL stencil[7];

	int ia1 = -3; int ib1 = 3;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
	    REAL tmp_r, tmp_l; 

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

		if(flag != 0) tmp_r = OCFD_weno7_kernel_M(&stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}
//==============================================================================================================================================================//

__device__ REAL sign(REAL x1, REAL x2){
	if(x2 >=0){
		x1 = fabs(x1);
	}else{
		x1 = -fabs(x1);
	}
	return x1;
}

__device__ REAL minmod2(REAL x1, REAL x2){

	REAL minmod2 = 0.5*(sign(1.0, x1) + sign(1.0, x2))*fmin(fabs(x1),fabs(x2));

    return minmod2;
}

__device__ REAL minmod4(REAL x1, REAL x2, REAL x3, REAL x4){

	REAL minmod4 = 0.5*(sign(1.0, x1) + sign(1.0, x2));

	minmod4 = minmod4*fabs(0.5*(sign(1.0, x1) + sign(1.0, x3)));
	minmod4 = minmod4*fabs(0.5*(sign(1.0, x1) + sign(1.0, x4)));

	REAL tmp  = fmin(x1, x2);
	REAL tmp1 = fmin(x3, x4);
	tmp = fmin(tmp,tmp1);

	minmod4 = minmod4*tmp;

    return minmod4;
}

//===================================================================2order_NND========================================================================//
__device__ REAL OCFD_NND2_kernel_P(REAL *stencil){

	REAL tmp = stencil[1] + 0.5*minmod2(stencil[2] - stencil[1], stencil[1] - stencil[0]);

	return tmp;
}


__global__ void OCFD_NND2_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[5];

	int ia1 = -2; int ib1 = 2;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
		REAL tmp_r, tmp_l;

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job); 

		if(flag != 0) tmp_r = OCFD_NND2_kernel_P(&stencil[1]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}


//----------------------------------------------------------------------------------------------------------------------------
__device__ REAL OCFD_NND2_kernel_M(REAL *stencil){

	REAL tmp = stencil[1] + 0.5*minmod2(stencil[0] - stencil[1], stencil[1] - stencil[2]);

	return tmp;
}


__global__ void OCFD_NND2_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[5];

	int ia1 = -2; int ib1 = 2;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
	    REAL tmp_r, tmp_l; 

		flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

		if(flag != 0) tmp_r = OCFD_NND2_kernel_M(&stencil[1]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}
//==================================================================================================================================


//===================================================================2order_NND========================================================================//
__device__ REAL OCFD_UP7_kernel_P(REAL *stencil){

	//REAL tmp = (3.0*stencil[0] - 28.0*stencil[1] + 126.0*stencil[2] - 420.0*stencil[3] + 105.0*stencil[4]
	//		+ 252.0*stencil[5] - 42.0*stencil[6] + 4.0*stencil[7])/420.0;
	REAL tmp = (-3.0*stencil[0] + 25.0*stencil[1] - 101.0*stencil[2] + 319.0*stencil[3] + 214.0*stencil[4]
			- 38.0*stencil[5] + 4.0*stencil[6])/420.0;

	return tmp;
}


__global__ void OCFD_UP7_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[8];

	int ia1 = -3; int ib1 = 4;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
		REAL tmp_r, tmp_l;

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job); 

		if(flag != 0) tmp_r = OCFD_UP7_kernel_P(&stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}


//----------------------------------------------------------------------------------------------------------------------------
__device__ REAL OCFD_UP7_kernel_M(REAL *stencil){

	//REAL tmp = -(3.0*stencil[7] - 28.0*stencil[6] + 126.0*stencil[5] - 420.0*stencil[4] + 105.0*stencil[3]
	//	+ 252.0*stencil[2] - 42.0*stencil[1] + 4.0*stencil[0])/420.0;
	REAL tmp = (-3.0*stencil[7] + 25.0*stencil[6] - 101.0*stencil[5] + 319.0*stencil[4] + 214.0*stencil[3]
		- 38.0*stencil[2] + 4.0*stencil[1])/420.0;

	return tmp;
}


__global__ void OCFD_UP7_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[8];

	int ia1 = -4; int ib1 = 3;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
	    REAL tmp_r, tmp_l; 

		flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

		if(flag != 0) tmp_r = OCFD_UP7_kernel_M(&stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}
//==================================================================================================================================


//===========================================================OMP6===================================================================
__device__ REAL OCFD_OMP6_kernel_P(int OMP6_FLAG, REAL *stencil){

	REAL m,n;

	if(OMP6_FLAG == 1){
		m = 0.001; n = 0.0;
	}else if(OMP6_FLAG == 2){
		m = 0.0; n = - 1.0/140.0;
	}else{
		m = 0.015; n = 0.0;
	}

	REAL mid_nf = 0.5*(m + n); m = 0.5*(m - n); 

	mid_nf =             ( 60.0*mid_nf  * stencil[7] + 
	  ( 1.0 -   60.0*m -  360.0*mid_nf) * stencil[6] + 
	  (-8.0 +  360.0*m +  900.0*mid_nf) * stencil[5] + 
	  (37.0 -  900.0*m - 1200.0*mid_nf) * stencil[4] + 
	  (37.0 + 1200.0*m +  900.0*mid_nf) * stencil[3] + 
	  (-8.0 -  900.0*m -  360.0*mid_nf) * stencil[2] + 
	  (1.0  +  360.0*m +   60.0*mid_nf) * stencil[1] - 
							     60.0*m * stencil[0])/60.0;

	m = stencil[3] + minmod2((stencil[4] - stencil[3]), (stencil[3] - stencil[2]));

	if((mid_nf - stencil[3])*(mid_nf - m) >= 1.e-10){
		REAL tmp, tmp1;

		m = stencil[2] + stencil[4] - 2.0*stencil[3];
		n = stencil[3]  + stencil[5] - 2.0*stencil[4];
		tmp  = 4.0*m - n;
		tmp1 = 4.0*n - m;

		tmp = 0.5*(stencil[3] + stencil[4]) - 0.5*minmod4(tmp, tmp1, n, m);

		n = stencil[1] + stencil[3] - 2.0*stencil[2];
		tmp1 = stencil[3] + 0.5*(stencil[3] - stencil[2]) + 4.0*minmod4(4*n - m, 4*m - n, m, n)/3.0;

		{
			m = fmin(stencil[3], stencil[4]);
			m = fmin(m, tmp);

			n = stencil[3] + 4.0*(stencil[3] - stencil[2]);

			n = fmin(stencil[3], n);
			n = fmin(n, tmp1);

			m = fmax(m, n);
		}
		{
			tmp = fmax(stencil[3], tmp);
			tmp = fmax(stencil[4], tmp);

			n = stencil[3] + 4.0*(stencil[3] - stencil[2]);

			n = fmax(stencil[3], n);
			n = fmax(n, tmp1);

			n = fmin(tmp, n);
		}

		mid_nf = mid_nf + minmod2(n - mid_nf, m - mid_nf);
	}

	return mid_nf;
}


__global__ void OCFD_OMP6_P_kernel(int i, int OMP6_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[8];

	int ia1 = -3; int ib1 = 4;
	
	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
		REAL tmp_r, tmp_l;

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

		if(flag != 0) tmp_r = OCFD_OMP6_kernel_P(OMP6_FLAG, &stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}
//---------------------------------------------------------OMP6_M---------------------------------------------------------------

__device__ REAL OCFD_OMP6_kernel_M(int OMP6_FLAG, REAL *stencil){

	REAL m,n;

	if(OMP6_FLAG == 1){
		m = 0.001; n = 0.0;
	}else if(OMP6_FLAG == 2){
		m = 0.0; n = - 1.0/140.0;
	}else{
		m = 0.015; n = 0.0;
	}

	REAL mid_nf = 0.5*(m + n); m = 0.5*(m - n); 

	mid_nf =            (  60.0*mid_nf  * stencil[0] + 
	   (1.0 -   60.0*m -  360.0*mid_nf) * stencil[1] + 
	  (-8.0 +  360.0*m +  900.0*mid_nf) * stencil[2] + 
	  (37.0 -  900.0*m - 1200.0*mid_nf) * stencil[3] + 
	  (37.0 + 1200.0*m +  900.0*mid_nf) * stencil[4]  + 
	  (-8.0 -  900.0*m -  360.0*mid_nf) * stencil[5]  + 
	   (1.0 +  360.0*m +   60.0*mid_nf) * stencil[6]  - 
							     60.0*m * stencil[7])/60.0;

	m = stencil[4] + minmod2((stencil[3] - stencil[4]), (stencil[4] - stencil[5]));

	if((mid_nf - stencil[4])*(mid_nf - m) >= 1.e-10){
		REAL tmp, tmp1;

		m = stencil[5] + stencil[3] - 2.0*stencil[4];
		n = stencil[4] + stencil[2] - 2.0*stencil[3];
		tmp  = 4.0*m - n;
		tmp1 = 4.0*n - m;

		tmp = 0.5*(stencil[4] + stencil[3]) - 0.5*minmod4(tmp, tmp1, n, m);

		n = stencil[6] + stencil[4] - 2.0*stencil[5];
		tmp1 = stencil[4] + 0.5*(stencil[4] - stencil[3]) + 4.0*minmod4(4*n - m, 4*m - n, m, n)/3.0;

		{
			m = fmin(stencil[5], stencil[3]);
			m = fmin(m, tmp);

			n = stencil[4] + 4.0*(stencil[4] - stencil[5]);

			n = fmin(stencil[4], n);
			n = fmin(n, tmp1);

			m = fmax(m, n);
		}
		{
			tmp = fmax(stencil[4], tmp);
			tmp = fmax(stencil[3], tmp);

			n = stencil[4] + 4.0*(stencil[4] - stencil[5]);

			n = fmax(stencil[4], n);
			n = fmax(n, tmp1);

			n = fmin(tmp, n);
		}

		mid_nf = mid_nf + minmod2(n - mid_nf, m - mid_nf);
	}

	return mid_nf;
}


__global__ void OCFD_OMP6_M_kernel(int i, int OMP6_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[8];

	int ia1 = -4; int ib1 = 3;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
	    REAL tmp_r, tmp_l; 

		flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job); 

		if(flag != 0) tmp_r = OCFD_OMP6_kernel_M(OMP6_FLAG, &stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}


__device__ REAL OCFD_weno5_kernel_P(REAL *stencil){
	//-2 ---- 1

	REAL S0 = 0.0, S1 = 0.0, S2 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp =     stencil[2] - 2.0*stencil[3] + stencil[4]; S0 += 13*tmp*tmp;
	tmp = 3.0*stencil[2] - 4.0*stencil[3] + stencil[4]; S0 +=  3*tmp*tmp;

	REAL q03 = (2.0*stencil[2] + 5.0*stencil[3] - stencil[4]);


	tmp = stencil[1] - 2.0*stencil[2] + stencil[3]; S1 += 13*tmp*tmp;
	tmp =                  stencil[1] - stencil[3]; S1 +=  3*tmp*tmp;

	REAL q13 = (-stencil[1] + 5.0*stencil[2] + 2.0*stencil[3]);


	tmp = stencil[0] - 2.0*stencil[1] +     stencil[2]; S2 += 13*tmp*tmp;
	tmp = stencil[0] - 4.0*stencil[1] + 3.0*stencil[2]; S2 +=  3*tmp*tmp;

	REAL q23 = (2.0*stencil[0] - 7.0*stencil[1] + 11.0*stencil[2]);


	REAL a0 = 3.0*((12.0*ep + S1)*(12.0*ep + S1))*((12.0*ep + S2)*(12.0*ep + S2));
	REAL a1 = 6.0*((12.0*ep + S0)*(12.0*ep + S0))*((12.0*ep + S2)*(12.0*ep + S2));
	REAL a2 =     ((12.0*ep + S0)*(12.0*ep + S0))*((12.0*ep + S1)*(12.0*ep + S1));
	
	
	tmp = (a0*q03 + a1*q13 + a2*q23)/(6.0*(a0 + a1 + a2));

	return tmp;
}


__global__ void OCFD_weno5_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[5];

	int ia1 = -2; int ib1 = 2;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
		REAL tmp_r, tmp_l;

		flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job); 

		if(flag != 0) tmp_r = OCFD_weno5_kernel_P(&stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_p_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);

	}
}


__device__ REAL OCFD_weno5_kernel_M(REAL *stencil){
	//-1  ----- 2

	REAL S0 = 0.0, S1 = 0.0, S2 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp =     stencil[2] - 2.0*stencil[1] + stencil[0]; S0 += 13*tmp*tmp;
	tmp = 3.0*stencil[2] - 4.0*stencil[1] + stencil[0]; S0 +=  3*tmp*tmp;

	REAL q03 = (2.0*stencil[2] + 5.0*stencil[1] - stencil[0]);


	tmp = stencil[3] - 2.0*stencil[2] + stencil[1]; S1 += 13*tmp*tmp;
	tmp =                  stencil[3] - stencil[1]; S1 +=  3*tmp*tmp;

	REAL q13 = (-stencil[3] + 5.0*stencil[2] + 2.0*stencil[1]);


	tmp = stencil[4] - 2.0*stencil[3] +     stencil[2]; S2 += 13*tmp*tmp;
	tmp = stencil[4] - 4.0*stencil[3] + 3.0*stencil[2]; S2 +=  3*tmp*tmp;

	REAL q23 = (2.0*stencil[4] - 7.0*stencil[3] + 11.0*stencil[2]);


	REAL a0 = 3.0*((12.0*ep + S1)*(12.0*ep + S1))*((12.0*ep + S2)*(12.0*ep + S2));
	REAL a1 = 6.0*((12.0*ep + S0)*(12.0*ep + S0))*((12.0*ep + S2)*(12.0*ep + S2));
	REAL a2 =     ((12.0*ep + S0)*(12.0*ep + S0))*((12.0*ep + S1)*(12.0*ep + S1));	
	
	tmp = (a0*q03 + a1*q13 + a2*q23)/(6.0*(a0 + a1 + a2));

	return tmp;
}


__global__ void OCFD_weno5_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job){
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[5];

	int ia1 = -2; int ib1 = 2;

	int flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[0], ia1, ib1, sort, job);

	if(flag != 0){
	    REAL tmp_r, tmp_l; 

		flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[0], ia1, ib1, job);  

		if(flag != 0) tmp_r = OCFD_weno5_kernel_M(&stencil[0]);

		tmp_l = __shfl_up_double(tmp_r, 1, warpSize);

		if(threadIdx.x != 0) put_du_m_kernel(flagxyzb, coords, tmp_r, tmp_l, du, i, Ajac, job);
	}
}

//-------------------------------------------------------------- CD6----------------------------------------------------------//
__global__ void OCFD_dx0_CD6_kernel(cudaField pf , cudaField pfx , cudaJobPackage job){
	// eyes on cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL x__3 = get_Field_LAP(pf, x-3, y, z);
		REAL x__2 = get_Field_LAP(pf, x-2, y, z);
		REAL x__1 = get_Field_LAP(pf, x-1, y, z);
		REAL  x_1 = get_Field_LAP(pf, x+1, y, z);
		REAL  x_2 = get_Field_LAP(pf, x+2, y, z);
		REAL  x_3 = get_Field_LAP(pf, x+3, y, z);
    
		get_Field(pfx , x-LAP,y-LAP,z-LAP) = (
				45.0*( x_1 - x__1 ) 
				-9.0*( x_2 - x__2 )
					+( x_3 - x__3 ) )
			/(60.0*hx_d);
	}
}




__global__ void OCFD_dy0_CD6_kernel(cudaField pf , cudaField pfy , cudaJobPackage job){
	// eyes on cells WITH LAPs
    unsigned int x = blockDim.y * blockIdx.y + threadIdx.y + job.start.x;
    unsigned int y = blockDim.x * blockIdx.x + threadIdx.x + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL y__3 = get_Field_LAP(pf, x, y-3, z);
		REAL y__2 = get_Field_LAP(pf, x, y-2, z);
		REAL y__1 = get_Field_LAP(pf, x, y-1, z);
		REAL  y_1 = get_Field_LAP(pf, x, y+1, z);
		REAL  y_2 = get_Field_LAP(pf, x, y+2, z);
		REAL  y_3 = get_Field_LAP(pf, x, y+3, z);
    
		get_Field(pfy , x-LAP,y-LAP,z-LAP) = (
				45.0*( y_1 - y__1 ) 
				-9.0*( y_2 - y__2 )
					+( y_3 - y__3 ) )
			/(60.0*hy_d);
	}
}


__global__ void OCFD_dz0_CD6_kernel(cudaField pf , cudaField pfz , cudaJobPackage job){

    unsigned int x = blockDim.y * blockIdx.y + threadIdx.y + job.start.x;
    unsigned int y = blockDim.z * blockIdx.z + threadIdx.z + job.start.y;
	unsigned int z = blockDim.x * blockIdx.x + threadIdx.x + job.start.z;

    if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL z__3 = get_Field_LAP(pf, x, y, z-3);
		REAL z__2 = get_Field_LAP(pf, x, y, z-2);
		REAL z__1 = get_Field_LAP(pf, x, y, z-1);
		REAL  z_1 = get_Field_LAP(pf, x, y, z+1);
		REAL  z_2 = get_Field_LAP(pf, x, y, z+2);
		REAL  z_3 = get_Field_LAP(pf, x, y, z+3);

        get_Field(pfz, x-LAP, y-LAP, z-LAP) = (
            45.0*( z_1 - z__1 ) 
            -9.0*( z_2 - z__2 )
                +( z_3 - z__3 ) )
            /(60.0*hz_d);
    }
}

//__global__ void OCFD_dz0_CD6_kernel(cudaField pf , cudaField pfz , cudaJobPackage job){
//	// eyes on cells WITH LAPs
//	extern __shared__ REAL hh[];
//	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
//	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
//	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
//	unsigned int id = threadIdx.z + (blockDim.z + 6)* (threadIdx.x + threadIdx.y * blockDim.x);
//
//	if(threadIdx.z < 6)	hh[id] = get_Field_LAP(pf, x, y, z-3);
//
//	if(x < job.end.x && y < job.end.y && z < job.end.z){
//
//		hh[id + 6] = get_Field_LAP(pf, x, y, z+3);
//		__syncthreads();
//
//		get_Field(pfz , x-LAP,y-LAP,z-LAP) = (
//				45.0*( hh[id + 4] - hh[id + 2] ) 
//				-9.0*( hh[id + 5] - hh[id + 1] )
//					+( hh[id + 6] - hh[id    ] ) )
//			/(60.0*hz_d);
//	}
//}
//===================================================================================================================================//


//-----------------------------------------------------------------CD8---------------------------------------------------------------//
__global__ void OCFD_dx0_CD8_kernel(cudaField pf , cudaField pfx , cudaJobPackage job){
	// eyes on cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL x__4 = get_Field_LAP(pf, x-4, y, z);
		REAL x__3 = get_Field_LAP(pf, x-3, y, z);
		REAL x__2 = get_Field_LAP(pf, x-2, y, z);
		REAL x__1 = get_Field_LAP(pf, x-1, y, z);
		REAL  x_1 = get_Field_LAP(pf, x+1, y, z);
		REAL  x_2 = get_Field_LAP(pf, x+2, y, z);
		REAL  x_3 = get_Field_LAP(pf, x+3, y, z);
		REAL  x_4 = get_Field_LAP(pf, x+4, y, z);
    
		get_Field(pfx, x-LAP, y-LAP, z-LAP) = (
				672.*( x_1 - x__1 ) 
				-168*( x_2 - x__2 )
				+32.*( x_3 - x__3 ) 
				  -3*( x_4 - x__4 ) )
			/(840.0*hx_d);
	}
}


__global__ void OCFD_dy0_CD8_kernel(cudaField pf , cudaField pfy , cudaJobPackage job){
	// eyes on cells WITH LAPs
    unsigned int x = blockDim.y * blockIdx.y + threadIdx.y + job.start.x;
    unsigned int y = blockDim.x * blockIdx.x + threadIdx.x + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL y__4 = get_Field_LAP(pf, x, y-4, z);
		REAL y__3 = get_Field_LAP(pf, x, y-3, z);
		REAL y__2 = get_Field_LAP(pf, x, y-2, z);
		REAL y__1 = get_Field_LAP(pf, x, y-1, z);
		REAL  y_1 = get_Field_LAP(pf, x, y+1, z);
		REAL  y_2 = get_Field_LAP(pf, x, y+2, z);
		REAL  y_3 = get_Field_LAP(pf, x, y+3, z);
		REAL  y_4 = get_Field_LAP(pf, x, y+4, z);
    
		get_Field(pfy , x-LAP,y-LAP,z-LAP) = (
				672.0*( y_1 - y__1 ) 
			   -168.0*( y_2 - y__2 )
				 +32.*( y_3 - y__3 ) 
				  -3.*( y_4 - y__4 ) )
			/(840.0*hy_d);
	}
}

__global__ void OCFD_dz0_CD8_kernel(cudaField pf , cudaField pfz , cudaJobPackage job){
	// eyes on cells WITH LAPs	
    unsigned int x = blockDim.y * blockIdx.y + threadIdx.y + job.start.x;
    unsigned int y = blockDim.z * blockIdx.z + threadIdx.z + job.start.y;
	unsigned int z = blockDim.x * blockIdx.x + threadIdx.x + job.start.z;

    if(x < job.end.x && y < job.end.y && z<job.end.z){
		REAL z__4 = get_Field_LAP(pf, x, y, z-4);
		REAL z__3 = get_Field_LAP(pf, x, y, z-3);
		REAL z__2 = get_Field_LAP(pf, x, y, z-2);
		REAL z__1 = get_Field_LAP(pf, x, y, z-1);
		REAL  z_1 = get_Field_LAP(pf, x, y, z+1);
		REAL  z_2 = get_Field_LAP(pf, x, y, z+2);
		REAL  z_3 = get_Field_LAP(pf, x, y, z+3);
		REAL  z_4 = get_Field_LAP(pf, x, y, z+4);

        get_Field(pfz , x-LAP,y-LAP,z-LAP) = (
                672.0*( z_1 - z__1 ) 
               -168.0*( z_2 - z__2 )
                 +32.*( z_3 - z__3 ) 
                  -3.*( z_4 - z__4 ) )
            /(840.0*hz_d);
	}
}

#ifdef __cplusplus
}
#endif

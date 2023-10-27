#include "parameters.h"
#include "utility.h"
#include "OCFD_time.h"

#include "cuda_commen.h"
#include "cuda_utility.h"
#include "parameters_d.h"


#ifdef __cplusplus
extern "C"{
#endif

// 旋转源项
// __global__ void source_kernel(cudaSoA du , cudaField rho , cudaField v, cudaField w, cudaField yy, cudaField zz, cudaJobPackage job)
// {
//     // eyes on no-lap region
//     unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
//     unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
//     unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

//     // TODO: define omega elsewhere
//     REAL Omega = 0.002;

//     if( x<job.end.x && y<job.end.y && z<job.end.z){

//         REAL y_ = get_Field_LAP(yy, x+LAP,y+LAP,z+LAP);
//         REAL z_ = get_Field_LAP(zz, x+LAP,y+LAP,z+LAP);
//         REAL rho_ = get_Field_LAP(rho, x+LAP,y+LAP,z+LAP);
//         REAL v_ = get_Field_LAP(v, x+LAP,y+LAP,z+LAP);
//         REAL w_ = get_Field_LAP(w, x+LAP,y+LAP,z+LAP);

//         REAL f_2 = rho_*(Omega*Omega * y_ + 2.0 * Omega * w_);
//         REAL f_3 = rho_*(Omega*Omega * z_ - 2.0 * Omega * v_);
//         REAL f_4 = rho_* Omega*Omega * (v_ * y_  + w_ * z_);

//         get_SoA(du , x,y,z , 2) += f_2;
//         get_SoA(du , x,y,z , 3) += f_3;
//         get_SoA(du , x,y,z , 4) += f_4;

//     }
// }

__global__ void OCFD_time_advance_ker1(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z){
		get_SoA(f , x,y,z , 0) = get_SoA(fn , x,y,z , 0) + dt_d*get_SoA(du , x,y,z , 0);
		get_SoA(f , x,y,z , 1) = get_SoA(fn , x,y,z , 1) + dt_d*get_SoA(du , x,y,z , 1);
		get_SoA(f , x,y,z , 2) = get_SoA(fn , x,y,z , 2) + dt_d*get_SoA(du , x,y,z , 2);
		get_SoA(f , x,y,z , 3) = get_SoA(fn , x,y,z , 3) + dt_d*get_SoA(du , x,y,z , 3);
		get_SoA(f , x,y,z , 4) = get_SoA(fn , x,y,z , 4) + dt_d*get_SoA(du , x,y,z , 4);

	}
}

__global__ void OCFD_time_advance_ker2(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL tmp1 = 3.0 / 4.0;
		REAL tmp2 = 1.0 / 4.0;

		get_SoA(f , x,y,z , 0) = tmp1*get_SoA(fn , x,y,z , 0) + tmp2*( get_SoA(f , x,y,z , 0) + dt_d*get_SoA(du , x,y,z , 0) );
		get_SoA(f , x,y,z , 1) = tmp1*get_SoA(fn , x,y,z , 1) + tmp2*( get_SoA(f , x,y,z , 1) + dt_d*get_SoA(du , x,y,z , 1) );
		get_SoA(f , x,y,z , 2) = tmp1*get_SoA(fn , x,y,z , 2) + tmp2*( get_SoA(f , x,y,z , 2) + dt_d*get_SoA(du , x,y,z , 2) );
		get_SoA(f , x,y,z , 3) = tmp1*get_SoA(fn , x,y,z , 3) + tmp2*( get_SoA(f , x,y,z , 3) + dt_d*get_SoA(du , x,y,z , 3) );
		get_SoA(f , x,y,z , 4) = tmp1*get_SoA(fn , x,y,z , 4) + tmp2*( get_SoA(f , x,y,z , 4) + dt_d*get_SoA(du , x,y,z , 4) );
	}
}

__global__ void OCFD_time_advance_ker3(cudaSoA f , cudaSoA fn , cudaSoA du , cudaSoA f_lap , cudaJobPackage job)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z){
		REAL tmp1 = 1.0 / 3.0;
		REAL tmp2 = 2.0 / 3.0;

		get_SoA_LAP(f_lap , x+LAP,y+LAP,z+LAP , 0) = get_SoA(f , x,y,z , 0) = tmp1*get_SoA(fn , x,y,z , 0) + tmp2*( get_SoA(f , x,y,z , 0) + dt_d*get_SoA(du , x,y,z , 0) );
		get_SoA_LAP(f_lap , x+LAP,y+LAP,z+LAP , 1) = get_SoA(f , x,y,z , 1) = tmp1*get_SoA(fn , x,y,z , 1) + tmp2*( get_SoA(f , x,y,z , 1) + dt_d*get_SoA(du , x,y,z , 1) );
		get_SoA_LAP(f_lap , x+LAP,y+LAP,z+LAP , 2) = get_SoA(f , x,y,z , 2) = tmp1*get_SoA(fn , x,y,z , 2) + tmp2*( get_SoA(f , x,y,z , 2) + dt_d*get_SoA(du , x,y,z , 2) );
		get_SoA_LAP(f_lap , x+LAP,y+LAP,z+LAP , 3) = get_SoA(f , x,y,z , 3) = tmp1*get_SoA(fn , x,y,z , 3) + tmp2*( get_SoA(f , x,y,z , 3) + dt_d*get_SoA(du , x,y,z , 3) );
		get_SoA_LAP(f_lap , x+LAP,y+LAP,z+LAP , 4) = get_SoA(f , x,y,z , 4) = tmp1*get_SoA(fn , x,y,z , 4) + tmp2*( get_SoA(f , x,y,z , 4) + dt_d*get_SoA(du , x,y,z , 4) );
	}
}

void OCFD_time_advance(int KRK)
{
	dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx,ny,nz);
    cudaJobPackage job( dim3(0,0,0) , dim3(nx,ny,nz) );

	// The RHS source is added here for every RK step
	// CUDA_LAUNCH(( source_kernel<<<griddim , blockdim>>>(*pdu_d, *pd_d, *pv_d, *pw_d, *pAyy_d, *pAzz_d, job)));

	switch (KRK)
	{
		case 1:
		{
			CUDA_LAUNCH(( OCFD_time_advance_ker1<<<griddim , blockdim>>>(*pf_d , *pfn_d , *pdu_d , job) ));
			break;
		}
		case 2:
		{
			CUDA_LAUNCH(( OCFD_time_advance_ker2<<<griddim , blockdim>>>(*pf_d , *pfn_d , *pdu_d , job) ));
			break;
		}
		case 3:
		{
			CUDA_LAUNCH(( OCFD_time_advance_ker3<<<griddim , blockdim>>>(*pf_d , *pfn_d , *pdu_d , *pf_lap_d , job) ));
			break;
		}
	}
}


#ifdef __cplusplus
}
#endif
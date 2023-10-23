//      boundary scheme
#include "parameters.h"
#include "utility.h"
#include "OCFD_Schemes.h"
#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "mpi.h"



#define PREPARE_x \
dim3 blockdim , griddim;\
cal_grid_block_dim(&griddim , &blockdim , blockdim_in.x , blockdim_in.y , blockdim_in.z , 1 , size.y , size.z);\


#define PREPARE_y \
dim3 blockdim , griddim;\
cal_grid_block_dim(&griddim , &blockdim , blockdim_in.x , blockdim_in.y , blockdim_in.z , size.x , 1 , size.z);\


#define PREPARE_z \
dim3 blockdim , griddim;\
cal_grid_block_dim(&griddim , &blockdim , blockdim_in.x , blockdim_in.y , blockdim_in.z , size.x , size.y , 1);\


#ifdef DEBUG_MODE
#define CHECK_SIZE(dir , call)\
if(size.dir >= LAP){\
	PREPARE_ ##dir\
	call;\
}else{\
	printf("job_in.start." #dir " = %d , job_in.size." #dir " = %d\n",job_in.start.dir , size.dir);\
	printf("illegal size , to launch %s , size." #dir " >= LAP (%d) is required\n" , __FUNCTION__ ,LAP);\
	MPI_Abort(MPI_COMM_WORLD , 1);\
}
#else
#define CHECK_SIZE(dir , call) \
PREPARE_ ##dir\
call;
#endif


#define CHECK_X(callm , callp)\
dim3 size;\
jobsize(&job_in , &size);\
if(npx == 0 && job_in.start.x == LAP){\
	CHECK_SIZE(x , callm)\
}\
if(npx == NPX0-1 && (job_in.start.x + size.x == nx_lap) ){\
	CHECK_SIZE(x , callp)\
}

#define CHECK_Y(callm , callp)\
dim3 size;\
jobsize(&job_in , &size);\
if(npy == 0 && job_in.start.y == LAP){\
	CHECK_SIZE(y , callm)\
}\
if(npy == NPY0-1 && (job_in.start.y + size.y == ny_lap) ){\
	CHECK_SIZE(y , callp)\
}

#define CHECK_Z(callm , callp)\
dim3 size;\
jobsize(&job_in , &size);\
if(npz == 0 && job_in.start.z == LAP){\
	CHECK_SIZE(z , callm)\
}\
if(npz == NPZ0-1 && (job_in.start.z + size.z == nz_lap) ){\
	CHECK_SIZE(z , callp)\
}

#ifdef __cplusplus
extern "C"{
#endif


#define D0bound1(coords)\
if(coords == 0){\
	*tmp = (stencil[-ka1+1] - stencil[-ka1]);\
	return 0;\
}else if(coords == 1){\
	*tmp = (stencil[-ka1+1] - stencil[-ka1-1])*0.5;\
	return 0;\
}else if(coords == 2){\
	*tmp = (stencil[-ka1-2] - 8.0*stencil[-ka1-1] + 8.0*stencil[-ka1+1] - stencil[-ka1+2])/12.0;\
	return 0;\
}else if(coords == 3){\
	*tmp = (stencil[-ka1+3] - stencil[-ka1-3]\
	  -9.0*(stencil[-ka1+2] - stencil[-ka1-2])\
	 +45.0*(stencil[-ka1+1] - stencil[-ka1-1]))/60.0;\
	 return 0;\
}\


#define D0bound2(coords)\
if(coords == -1){\
	*tmp = (stencil[-ka1] - stencil[-ka1-1]);\
	return 0;\
}else if(coords == -2){\
	*tmp = (stencil[-ka1+1] - stencil[-ka1-1])*0.5;\
	return 0;\
}else if(coords == -3){\
	*tmp = (stencil[-ka1-2] - 8.0*stencil[-ka1-1] + 8.0*stencil[-ka1+1] - stencil[-ka1+2])/12.0;\
	return 0;\
}else if(coords == -4){\
	*tmp = (stencil[-ka1+3] - stencil[-ka1-3]\
	  -9.0*(stencil[-ka1+2] - stencil[-ka1-2])\
	 +45.0*(stencil[-ka1+1] - stencil[-ka1-1]))/60.0;\
	 return 0;\
}\


// =========================================================================================================== //
__device__ int OCFD_D0bound_scheme_kernel(REAL* tmp, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, cudaJobPackage job){
    int tmp1;

	switch(flagxyzb.y){
		case 1:
		{
            D0bound1(coords.x)
		}
        break;

		case 2:
		{
            D0bound1(coords.y)
		}
        break;

		case 3:
		{
            D0bound1(coords.z)
		}
        break;

		case 4:
		{
            tmp1 = coords.x + job.start.x - job.end.x;
            D0bound2(tmp1)
		}
        break;

		case 5:
		{
            tmp1 = coords.y + job.start.y - job.end.y;
            D0bound2(tmp1)
		}
        break;

		case 6:
		{
            tmp1 = coords.z + job.start.z - job.end.z;
            D0bound2(tmp1)
		}
        break;

		case 7:
		{
            D0bound1(coords.x)
            tmp1 = coords.x + job.start.x - job.end.x;
            D0bound2(tmp1)
		}
        break;

		case 8:
		{
            D0bound1(coords.y)
            tmp1 = coords.y + job.start.y - job.end.y;
            D0bound2(tmp1)
		}
        break;

		case 9:
		{
            D0bound1(coords.z)
            tmp1 = coords.z + job.start.z - job.end.z;
            D0bound2(tmp1)
		}
        break;
	}

	return 1;

}


__global__ void OCFD_Dx0_bound_kernel_m(cudaField f , cudaField fx , cudaJobPackage job){
	// eyes on cells WITHOUT LAP
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(y < job.end.y && z < job.end.z){
		// 0
		get_Field(fx , 0 , y-LAP, z-LAP) = ( get_Field_LAP(f , LAP+1 , y , z) - get_Field_LAP(f , LAP , y , z))/hx_d;
		get_Field(fx , 1 , y-LAP, z-LAP) = ( get_Field_LAP(f , LAP+2 , y , z) - get_Field_LAP(f , LAP , y , z))*0.5/hx_d;

		get_Field(fx , 2 , y-LAP, z-LAP) = ( get_Field_LAP(f , LAP   , y , z) - 8.0*get_Field_LAP(f , LAP+1 , y , z) 
								       + 8.0*get_Field_LAP(f , LAP+3 , y , z) -     get_Field_LAP(f , LAP+4 , y , z))/(12.0*hx_d);

		get_Field(fx , 3 , y-LAP , z-LAP) = ( get_Field_LAP(f , LAP+6 , y , z) - get_Field_LAP(f , LAP   , y , z)
		 						        -9.0*(get_Field_LAP(f , LAP+5 , y , z) - get_Field_LAP(f , LAP+1 , y , z) )
								       +45.0*(get_Field_LAP(f , LAP+4 , y , z) - get_Field_LAP(f , LAP+2 , y , z)) )/(60.0*hx_d);

	}
}


__global__ void OCFD_Dx0_bound_kernel_p(cudaField f , cudaField fx , cudaJobPackage job){
	// eyes on cells WITHOUT LAP
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(y < job.end.y && z < job.end.z){
		unsigned int tmp = nx_d+LAP-1;
		get_Field(fx , nx_d - 1 , y-LAP , z-LAP) = ( get_Field_LAP(f , tmp ,   y , z) -     get_Field_LAP(f , tmp-1 , y , z))/hx_d;
		get_Field(fx , nx_d - 2 , y-LAP , z-LAP) = ( get_Field_LAP(f , tmp ,   y , z) -     get_Field_LAP(f , tmp-2 , y , z))*0.5/hx_d;

		get_Field(fx , nx_d - 3 , y-LAP , z-LAP) = ( get_Field_LAP(f , tmp-4 , y , z) - 8.0*get_Field_LAP(f , tmp-3 , y , z) 
										       + 8.0*get_Field_LAP(f , tmp-1 , y , z) -     get_Field_LAP(f , tmp   , y , z))/(12.0*hx_d);

		get_Field(fx , nx_d - 4 , y-LAP , z-LAP) = ( get_Field_LAP(f , tmp     , y , z) - get_Field_LAP(f , tmp-6     , y , z)
		 								      -9.0*( get_Field_LAP(f , tmp - 1 , y , z) - get_Field_LAP(f , tmp - 5 , y , z) )
									         +45.0*( get_Field_LAP(f , tmp - 2 , y , z) - get_Field_LAP(f , tmp - 4 , y , z)) )/(60.0*hx_d);

	}
}

// =========================================================================================================== //

__global__ void OCFD_Dy0_bound_kernel_m(cudaField f , cudaField fx , cudaJobPackage job){
	// eyes on cells WITHOUT LAP
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(z < job.end.z && x < job.end.x){
		get_Field(fx , x-LAP, 0, z-LAP) = ( get_Field_LAP(f , x, LAP+1, z) - get_Field_LAP(f ,x, LAP, z))/hy_d;
		get_Field(fx , x-LAP, 1, z-LAP) = ( get_Field_LAP(f , x, LAP+2, z) - get_Field_LAP(f ,x, LAP, z))*0.5/hy_d;

		get_Field(fx , x-LAP, 2, z-LAP) = ( get_Field_LAP(f , x, LAP, z) - 8.0*get_Field_LAP(f, x, LAP+1, z) 
							          + 8.0*get_Field_LAP(f , x, LAP+3, z) -     get_Field_LAP(f, x, LAP+4, z))/(12.0*hy_d);

		get_Field(fx , x-LAP, 3, z-LAP) = ( get_Field_LAP(f , x, LAP+6, z) - get_Field_LAP(f , x, LAP, z)
		 					          -9.0*(get_Field_LAP(f , x, LAP+5, z) - get_Field_LAP(f , x, LAP+1, z) )
							         +45.0*(get_Field_LAP(f , x, LAP+4, z) - get_Field_LAP(f , x, LAP+2, z)) )/(60.0*hy_d);

	}
}

__global__ void OCFD_Dy0_bound_kernel_p(cudaField f , cudaField fx , cudaJobPackage job){
	// eyes on cells WITHOUT LAP
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(z < job.end.z && x < job.end.x){
		unsigned int tmp = ny_d+LAP-1;
		get_Field(fx, x-LAP, ny_d-1, z-LAP) = -( get_Field_LAP(f , x, tmp-1, z) - get_Field_LAP(f ,x, tmp, z))/hy_d;
		get_Field(fx, x-LAP, ny_d-2, z-LAP) = -( get_Field_LAP(f , x, tmp-2, z) - get_Field_LAP(f ,x, tmp, z))*0.5/hy_d;

		get_Field(fx, x-LAP, ny_d-3, z-LAP) = -( get_Field_LAP(f , x, tmp, z) - 8.0*get_Field_LAP(f, x, tmp-1, z) 
							          + 8.0*get_Field_LAP(f , x, tmp-3, z) -     get_Field_LAP(f, x, tmp-4, z))/(12.0*hy_d);

		get_Field(fx, x-LAP, ny_d-4, z-LAP) = -( get_Field_LAP(f , x, tmp-6, z) - get_Field_LAP(f , x, tmp, z)
		 					          -9.0*(get_Field_LAP(f , x, tmp-5, z) - get_Field_LAP(f , x, tmp-1, z) )
							         +45.0*(get_Field_LAP(f , x, tmp-4, z) - get_Field_LAP(f , x, tmp-2, z)) )/(60.0*hy_d);

	}
}

// =========================================================================================================== //


__global__ void OCFD_Dz0_bound_kernel_m(cudaField f , cudaField fx , cudaJobPackage job){
	// eyes on cells WITHOUT LAP
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;

	if(y < job.end.y && x < job.end.x){
		get_Field(fx , x-LAP,y-LAP, 0) = ( get_Field_LAP(f , x,y, LAP+1) - get_Field_LAP(f ,x,y, LAP))/hz_d;
		get_Field(fx , x-LAP,y-LAP, 1) = ( get_Field_LAP(f , x,y, LAP+2) - get_Field_LAP(f ,x,y, LAP))*0.5/hz_d;

		get_Field(fx , x-LAP,y-LAP, 2) = ( get_Field_LAP(f , x, y , LAP  ) - 8.0*get_Field_LAP(f , x,y, LAP+1) 
							   + 8.0*get_Field_LAP(f , x, y , LAP+3) -     get_Field_LAP(f , x,y, LAP+4))/(12.0*hz_d);

		get_Field(fx , x-LAP,y-LAP, 3) = ( get_Field_LAP(f , x,y, LAP+6) - get_Field_LAP(f , x,y, LAP)
		 					   -9.0*(get_Field_LAP(f , x,y, LAP+5) - get_Field_LAP(f , x,y, LAP+1) )
							  +45.0*(get_Field_LAP(f , x,y, LAP+4) - get_Field_LAP(f , x,y, LAP+2)) )/(60.0*hz_d);

	}
}

__global__ void OCFD_Dz0_bound_kernel_p(cudaField f , cudaField fx , cudaJobPackage job){
	// eyes on cells WITHOUT LAP
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;

	if(y < job.end.y && x < job.end.x){
		unsigned int tmp = nz_d+LAP-1;
		get_Field(fx ,  x-LAP,y-LAP, nz_d - 1)=  ( get_Field_LAP(f , x,y, tmp) -     get_Field_LAP(f , x,y, tmp-1))/hz_d;
		get_Field(fx ,  x-LAP,y-LAP, nz_d - 2) = ( get_Field_LAP(f , x,y, tmp) -     get_Field_LAP(f , x,y, tmp-2))*0.5/hz_d;

		get_Field(fx ,  x-LAP,y-LAP, nz_d - 3) = ( get_Field_LAP(f , x,y, tmp-4) - 8.0*get_Field_LAP(f , x,y, tmp-3) 
										      + 8.0*get_Field_LAP(f , x,y, tmp-1) -     get_Field_LAP(f , x,y, tmp  ))/(12.0*hz_d);

		get_Field(fx ,  x-LAP,y-LAP, nz_d - 4) = ( get_Field_LAP(f , x,y, tmp  ) - get_Field_LAP(f , x,y, tmp - 6)
		 									  -9.0*(get_Field_LAP(f , x,y, tmp-1) - get_Field_LAP(f , x,y, tmp - 5) )
												+45.0*(get_Field_LAP(f , x,y, tmp-2) - get_Field_LAP(f , x,y, tmp - 4)) )/(60.0*hz_d);

	}
}




void OCFD_Dx0_bound(cudaField f , cudaField fx , cudaJobPackage job_in , dim3 blockdim_in, cudaStream_t *stream){ 
	CHECK_X(
		{
			CUDA_LAUNCH(( OCFD_Dx0_bound_kernel_m<<<griddim , blockdim, 0, *stream>>>(f, fx, job_in) ));

		},{
			CUDA_LAUNCH(( OCFD_Dx0_bound_kernel_p<<<griddim , blockdim, 0, *stream>>>(f, fx, job_in) ));
		}
	)
}


void OCFD_Dy0_bound(cudaField f , cudaField fx , cudaJobPackage job_in , dim3 blockdim_in, cudaStream_t *stream){
	CHECK_Y(
		{
			CUDA_LAUNCH(( OCFD_Dy0_bound_kernel_m<<<griddim , blockdim, 0, *stream>>>(f, fx, job_in) ));
		},
		{
			CUDA_LAUNCH(( OCFD_Dy0_bound_kernel_p<<<griddim , blockdim, 0, *stream>>>(f, fx, job_in) ));
		}
	)
}


void OCFD_Dz0_bound(cudaField f , cudaField fx , cudaJobPackage job_in , dim3 blockdim_in, cudaStream_t *stream){
	CHECK_Z(
		{
			CUDA_LAUNCH(( OCFD_Dz0_bound_kernel_m<<<griddim , blockdim, 0, *stream>>>(f, fx, job_in) ));
		},
		{
			CUDA_LAUNCH(( OCFD_Dz0_bound_kernel_p<<<griddim , blockdim, 0, *stream>>>(f, fx, job_in) ));
		}
	)
}


void OCFD_bound_non_ref(dim3 *flagxyzb, int Non_ref, cudaJobPackage job){
	// eyes on field WITH LAPs
	dim3 size;
	jobsize(&job, &size);
	switch(flagxyzb->x){
		case 1:
        {
		    if(npx == 0 && job.start.x == LAP){
                flagxyzb->z = Non_ref; 
            }else{
                flagxyzb->z = 0; 
            }
        }
		break;

		case 2:
        {
		    if(npy == 0 && job.start.y == LAP){
                flagxyzb->z = Non_ref; 
            }else{
                flagxyzb->z = 0; 
            }
        }
		break;

		case 3:
        {
		    if(npz == 0 && job.start.z == LAP){
                flagxyzb->z = Non_ref; 
            }else{
                flagxyzb->z = 0; 
            }
        }
		break;

		case 4:
        {
		    if(npx == NPX0-1 && job.end.x == nx_lap){
                flagxyzb->z = Non_ref; 
            }else{
                flagxyzb->z = 0; 
            }
        }
		break;

		case 5:
        {
		    if(npy == NPY0-1 && job.end.y == ny_lap){
                flagxyzb->z = Non_ref; 
            }else{
                flagxyzb->z = 0; 
            }
        }
		break;

		case 6:
        {
		    if(npz == NPZ0-1 && job.end.z == nz_lap){ 
                flagxyzb->z = Non_ref; 
            }else{
                flagxyzb->z = 0; 
            }
        }
		break;
	}
}



void OCFD_bound(dim3 *flagxyzb, int boundp, int boundm, cudaJobPackage job){
	// eyes on field WITH LAPs
	dim3 size;
    int flag = 0;
	jobsize(&job, &size);
	switch(flagxyzb->x){
		case 1:
		case 4:
        {
		    if(npx == 0 && job.start.x == LAP && boundp == 1){
                flagxyzb->y = 1; 
                flag = 1;
            }
		    if(npx == NPX0-1 && job.end.x == nx_lap && boundm == 1){
                flagxyzb->y = 4;
                if(flag == 1) flagxyzb->y = 7;
            }
        }
		break;

		case 2:
		case 5:
        {
		    if(npy == 0 && job.start.y == LAP && boundp == 1){
                flagxyzb->y = 2;
                flag = 1;
            }
		    if(npy == NPY0-1 && job.end.y == ny_lap && boundm == 1){
                flagxyzb->y = 5;
                if(flag == 1) flagxyzb->y = 8;
            }
        }
		break;

		case 3:
		case 6:
        {
		    if(npz == 0 && job.start.z == LAP && boundp == 1){
                flagxyzb->y = 3;
                flag = 1;
            }
		    if(npz == NPZ0-1 && job.end.z == nz_lap && boundm == 1){ 
                flagxyzb->y = 6;
                if(flag == 1) flagxyzb->y = 9;
            }
        }
		break;
	}
}

/*__device__ int OCFD_bound_scheme_kernel_p(int flag, dim3 flagxyzb, dim3 coords, cudaSoA du, int num, cudaField fx, REAL *stencil, int ka1, int kb1, cudaJobPackage job){
	unsigned int offset_out = job.start.x + fx.pitch*(job.start.y + ny_d*job.start.z);
	if(flag != 0){
		switch(flagxyzb.x){
			case 4:
			if(threadIdx.x != 0) get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp_r - tmp_l)/hx_d;
			break;

			case 5:
			if(threadIdx.x != 0) get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp_r - tmp_l)/hy_d;
			break;

			case 6:
			if(threadIdx.x != 0) get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp_r - tmp_l)/hz_d;
			break;
		}

		switch(flagxyzb.y){
			case 4:
			{
				if(coords.x == job.end.x-job.start.x-1){
					REAL tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1] - stencil[-ka1-1]);
					REAL tmp2 = stencil[-ka1-1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1-1] - stencil[-ka1-2]);

					get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp - tmp2)/hx_d;

					return 0;

				}else if(coords.x >= job.end.x-job.start.x-kb1){
					REAL tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1+1] - stencil[-ka1], stencil[-ka1] - stencil[-ka1-1]);
					REAL tmp2 = stencil[-ka1-1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1-1] - stencil[-ka1-2]);

					get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp - tmp2)/hx_d;

					return 0;
				}
			}
			break;

			case 5:
			{
				if(coords.y == job.end.y-job.start.y-1){
					REAL tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1] - stencil[-ka1-1]);
					REAL tmp2 = stencil[-ka1-1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1-1] - stencil[-ka1-2]);

					get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp - tmp2)/hy_d;

					return 0;
				
				}else if(coords.y >= job.end.y-job.start.y-kb1){
					REAL tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1+1] - stencil[-ka1], stencil[-ka1] - stencil[-ka1-1]);
					REAL tmp2 = stencil[-ka1-1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1-1] - stencil[-ka1-2]);

					get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp - tmp2)/hy_d;

					return 0;
				}
			}
			break;

			case 6:
			{
				if(coords.z == job.end.z-job.start.z-1){
					REAL tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1] - stencil[-ka1-1]);
					REAL tmp2 = stencil[-ka1-1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1-1] - stencil[-ka1-2]);

					get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp - tmp2)/hz_d;

					return 0;
				
				}else if(coords.z >= job.end.z-job.start.z-kb1){
					REAL tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1+1] - stencil[-ka1], stencil[-ka1] - stencil[-ka1-1]);
					REAL tmp2 = stencil[-ka1-1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1-1] - stencil[-ka1-2]);

					get_Field(fx, coords.x-LAP, coords.y-LAP, coords.z-LAP, offset_out) = (tmp - tmp2)/hz_d;

					return 0;
				}
			}
			break;
		}
	}

	return flag;
}*/

__device__ REAL OCFD_weno5_kernel_P_right(REAL *stencil){

	REAL S2 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;


	tmp = stencil[0] - 2.0*stencil[1] +     stencil[2]; S2 += 13*tmp*tmp;
	tmp = stencil[0] - 4.0*stencil[1] + 3.0*stencil[2]; S2 +=  3*tmp*tmp;

	REAL a2 = 1.0/((12.0*ep + S2)*(12.0*ep + S2));
	REAL q23 = (2.0*stencil[0] - 7.0*stencil[1] + 11.0*stencil[2]);
	
	
	tmp = a2*q23/(6.0*a2);

	return tmp;
}


__device__ REAL OCFD_weno5_kernel_P_lift(REAL *stencil){

	REAL S0 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp =     stencil[2] - 2.0*stencil[3] + stencil[4]; S0 += 13*tmp*tmp;
	tmp = 3.0*stencil[2] - 4.0*stencil[3] + stencil[4]; S0 +=  3*tmp*tmp;

	REAL a0 = 1.0/((12.0*ep + S0)*(12.0*ep + S0));
	REAL q03 = (2.0*stencil[2] + 5.0*stencil[3] - stencil[4]);
	
	
	tmp = a0*q03/(6.0*a0);

	return tmp;
}


__device__ REAL OCFD_weno5_kernel_M_right(REAL *stencil){

	REAL S0 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp =     stencil[2] - 2.0*stencil[1] + stencil[0]; S0 += 13*tmp*tmp;
	tmp = 3.0*stencil[2] - 4.0*stencil[1] + stencil[0]; S0 +=  3*tmp*tmp;

	REAL a0 = 1.0/((12.0*ep + S0)*(12.0*ep + S0));
	REAL q03 = (2.0*stencil[2] + 5.0*stencil[1] - stencil[0]);
	
	
	tmp = a0*q03/(6.0*a0);

	return tmp;
}


__device__ REAL OCFD_weno5_kernel_M_lift(REAL *stencil){

	REAL S2 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;


	tmp = stencil[4] - 2.0*stencil[3] +     stencil[2]; S2 += 13*tmp*tmp;
	tmp = stencil[4] - 4.0*stencil[3] + 3.0*stencil[2]; S2 +=  3*tmp*tmp;

	REAL a2 = 1.0/((12.0*ep + S2)*(12.0*ep + S2));
	REAL q23 = (2.0*stencil[4] - 7.0*stencil[3] + 11.0*stencil[2]);
	
	
	tmp = a2*q23/(6.0*a2);

	return tmp;
}

__device__ REAL OCFD_weno5_kernel_P_right_plus(REAL *stencil){

	REAL S1 = 0.0, S2 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp = stencil[1] - 2.0*stencil[2] + stencil[3]; S1 += 13*tmp*tmp;
	tmp =                  stencil[1] - stencil[3]; S1 +=  3*tmp*tmp;

	REAL a1 = 6.0/((12.0*ep + S1)*(12.0*ep + S1));
	REAL q13 = (-stencil[1] + 5.0*stencil[2] + 2.0*stencil[3]);


	tmp = stencil[0] - 2.0*stencil[1] +     stencil[2]; S2 += 13*tmp*tmp;
	tmp = stencil[0] - 4.0*stencil[1] + 3.0*stencil[2]; S2 +=  3*tmp*tmp;

	REAL a2 = 1.0/((12.0*ep + S2)*(12.0*ep + S2));
	REAL q23 = (2.0*stencil[0] - 7.0*stencil[1] + 11.0*stencil[2]);
	
	
	tmp = (a1*q13 + a2*q23)/(6.0*(a1 + a2));

	return tmp;
}


__device__ REAL OCFD_weno5_kernel_P_lift_plus(REAL *stencil){

	REAL S0 = 0.0, S1 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp =     stencil[2] - 2.0*stencil[3] + stencil[4]; S0 += 13*tmp*tmp;
	tmp = 3.0*stencil[2] - 4.0*stencil[3] + stencil[4]; S0 +=  3*tmp*tmp;

	REAL a0 = 3.0/((12.0*ep + S0)*(12.0*ep + S0));
	REAL q03 = (2.0*stencil[2] + 5.0*stencil[3] - stencil[4]);


	tmp = stencil[1] - 2.0*stencil[2] + stencil[3]; S1 += 13*tmp*tmp;
	tmp =                  stencil[1] - stencil[3]; S1 +=  3*tmp*tmp;

	REAL a1 = 6.0/((12.0*ep + S1)*(12.0*ep + S1));
	REAL q13 = (-stencil[1] + 5.0*stencil[2] + 2.0*stencil[3]);
	
	
	tmp = (a0*q03 + a1*q13)/(6.0*(a0 + a1));

	return tmp;
}


__device__ REAL OCFD_weno5_kernel_M_right_plus(REAL *stencil){

	REAL S0 = 0.0, S1 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp =     stencil[2] - 2.0*stencil[1] + stencil[0]; S0 += 13*tmp*tmp;
	tmp = 3.0*stencil[2] - 4.0*stencil[1] + stencil[0]; S0 +=  3*tmp*tmp;

	REAL a0 = 3.0/((12.0*ep + S0)*(12.0*ep + S0));
	REAL q03 = (2.0*stencil[2] + 5.0*stencil[1] - stencil[0]);


	tmp = stencil[3] - 2.0*stencil[2] + stencil[1]; S1 += 13*tmp*tmp;
	tmp =                  stencil[3] - stencil[1]; S1 +=  3*tmp*tmp;

	REAL a1 = 6.0/((12.0*ep + S1)*(12.0*ep + S1));
	REAL q13 = (-stencil[3] + 5.0*stencil[2] + 2.0*stencil[1]);
	
	
	tmp = (a0*q03 + a1*q13)/(6.0*(a0 + a1));

	return tmp;
}


__device__ REAL OCFD_weno5_kernel_M_lift_plus(REAL *stencil){

	REAL S1 = 0.0, S2 = 0.0;
	REAL tmp;
	REAL ep = 1e-6;

	tmp = stencil[3] - 2.0*stencil[2] + stencil[1]; S1 += 13*tmp*tmp;
	tmp =                  stencil[3] - stencil[1]; S1 +=  3*tmp*tmp;

	REAL a1 = 6.0/((12.0*ep + S1)*(12.0*ep + S1));
	REAL q13 = (-stencil[3] + 5.0*stencil[2] + 2.0*stencil[1]);


	tmp = stencil[4] - 2.0*stencil[3] +     stencil[2]; S2 += 13*tmp*tmp;
	tmp = stencil[4] - 4.0*stencil[3] + 3.0*stencil[2]; S2 +=  3*tmp*tmp;

	REAL a2 = 1.0/((12.0*ep + S2)*(12.0*ep + S2));
	REAL q23 = (2.0*stencil[4] - 7.0*stencil[3] + 11.0*stencil[2]);
	
	
	tmp = (a1*q13 + a2*q23)/(6.0*(a1 + a2));

	return tmp;
}
//tmp = (2.0*stencil[-ka1+1] + 5.0*stencil[-ka1+2] + stencil[-ka1+3])/6.0;
//tmp = (11.0*stencil[-ka1] - 7.0*stencil[-ka1-1] + 2.0*stencil[-ka1-2])/6.0;

#define boundp1(coords)\
if(coords <= -ka1){\
	if(coords == 0){\
		*tmp = stencil[-ka1+1];\
	}\
	if(coords == 1){\
		*tmp = OCFD_weno5_kernel_P_lift(&stencil[-ka1-2]);\
	}\
	if(coords == 2){\
		*tmp = OCFD_weno5_kernel_P_lift_plus(&stencil[-ka1-2]);\
	}\
	if(coords == 3){\
		*tmp = OCFD_weno5_kernel_P(&stencil[-ka1-2]);\
	}\
	return 0;\
}\


#define boundp2(coords)\
if(coords > -kb1){\
	if(coords == -1){\
		*tmp = stencil[-ka1] + 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1] - stencil[-ka1-1]);\
	}\
	if(coords == -2){\
		*tmp = OCFD_weno5_kernel_P_right_plus(&stencil[-ka1-2]);\
	}\
	if(coords <= -3){\
		*tmp = OCFD_weno5_kernel_P(&stencil[-ka1-2]);\
	}\
	return 0;\
}\

__device__ int OCFD_bound_scheme_kernel_p(REAL* tmp, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, int kb1, cudaJobPackage job){
    int tmp1;

	switch(flagxyzb.y){
		case 1:
		{
            boundp1(coords.x)
		}
        break;

		case 2:
		{
            boundp1(coords.y)
		}
        break;

		case 3:
		{
            boundp1(coords.z)
		}
        break;

		case 4:
		{
            tmp1 = coords.x + job.start.x - job.end.x;
            boundp2(tmp1)
		}
        break;

		case 5:
		{
            tmp1 = coords.y + job.start.y - job.end.y;
            boundp2(tmp1)
		}
        break;

		case 6:
		{
            tmp1 = coords.z + job.start.z - job.end.z;
            boundp2(tmp1)
		}
        break;

		case 7:
		{
            boundp1(coords.x)
            tmp1 = coords.x + job.start.x - job.end.x;
            boundp2(tmp1)
		}
        break;

		case 8:
		{
            boundp1(coords.y)
            tmp1 = coords.y + job.start.y - job.end.y;
            boundp2(tmp1)
		}
        break;

		case 9:
		{
            boundp1(coords.z)
            tmp1 = coords.z + job.start.z - job.end.z;
            boundp2(tmp1)
		}
        break;
	}

	return 1;

}

#define boundm1(coords)\
if(coords < -ka1){\
	if(coords == 0){\
		*tmp = OCFD_weno5_kernel_M_lift(&stencil[-ka1-2]);\
	}\
	if(coords == 1){\
		*tmp = OCFD_weno5_kernel_M_lift_plus(&stencil[-ka1-2]);\
	}\
	if(coords >= 2){\
		*tmp = OCFD_weno5_kernel_M(&stencil[-ka1-2]);\
	}\
	return 0;\
}\

#define boundm2(coords)\
if(coords >= -kb1-1){\
	if(coords == -1){\
		*tmp = stencil[-ka1-1];\
	}\
	if(coords == -2){\
		*tmp = stencil[-ka1] - 0.5*minmod2(stencil[-ka1] - stencil[-ka1-1], stencil[-ka1] - stencil[-ka1-1]);\
	}\
	if(coords == -3){\
		*tmp = OCFD_weno5_kernel_M_right_plus(&stencil[-ka1-2]);\
	}\
	if(coords == -4){\
		*tmp = OCFD_weno5_kernel_M(&stencil[-ka1-2]);\
	}\
	return 0;\
}\

__device__ int OCFD_bound_scheme_kernel_m(REAL* tmp, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, int kb1, cudaJobPackage job){
    int tmp1;

	switch(flagxyzb.y){
		case 1:
		{
            boundm1(coords.x)
		}
        break;

		case 2:
		{
            boundm1(coords.y)
		}
        break;

		case 3:
		{
            boundm1(coords.z)
		}
        break;

		case 4:
		{
            tmp1 = coords.x + job.start.x - job.end.x;
            boundm2(tmp1)
		}
        break;

		case 5:
		{
            tmp1 = coords.y + job.start.y - job.end.y;
            boundm2(tmp1)
		}
        break;

		case 6:
		{
            tmp1 = coords.z + job.start.z - job.end.z;
            boundm2(tmp1)
		}
        break;
		
		case 7:
		{
            boundm1(coords.x)
            tmp1 = coords.x + job.start.x - job.end.x;
            boundm2(tmp1)
		}
        break;

		case 8:
		{
            boundm1(coords.y)
            tmp1 = coords.y + job.start.y - job.end.y;
            boundm2(tmp1)
		}
        break;

		case 9:
		{
            boundm1(coords.z)
            tmp1 = coords.z + job.start.z - job.end.z;
            boundm2(tmp1)
		}
        break;
	}

	return 1;

}


#ifdef __cplusplus
}
#endif

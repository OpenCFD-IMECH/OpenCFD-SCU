// Boundary condition for flow over a  3D Liftbody -------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "parameters.h"
#include "parameters_d.h"
#include "utility.h"
#include "io_warp.h"
#include "cuda_commen.h"
#include "cuda_utility.h"

//#include "OCFD_boundary_init.h"
#include "OCFD_init.h"

#ifdef __cplusplus
extern "C"{
#endif

extern cudaField *pu2d_inlet_d; //[5][nz][ny]
extern cudaField *pu2d_upper_d; //[5][ny][nx]
//extern cudaField *pv_dist_wall_d; // [ny][nx]
extern cudaField *pv_dist_coeff_d; // [3][ny][nx]
extern cudaField *pu_dist_upper_d; // [ny][nx]

extern const char v_dist_need;
extern const char TW_postive;

extern REAL *fait;
extern REAL *TM;

__global__ void do_u2d_inlet_kernel(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField inlet , cudaJobPackage job){
	// with LAPs
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	if(y < job.end.y && z < job.end.z){
		unsigned int ylap = y-LAP;
		unsigned int zlap = z-LAP;
		for(int i = 0; i <= LAP; i++){
			get_Field_LAP(d, i, y, z) = *(inlet.ptr + ylap + inlet.pitch *  zlap);
			get_Field_LAP(u, i, y, z) = *(inlet.ptr + ylap + inlet.pitch *( zlap + 1*nz_d) );
			get_Field_LAP(v, i, y, z) = *(inlet.ptr + ylap + inlet.pitch *( zlap + 2*nz_d) );
			get_Field_LAP(w, i, y, z) = *(inlet.ptr + ylap + inlet.pitch *( zlap + 3*nz_d) );
			get_Field_LAP(T, i, y, z) = *(inlet.ptr + ylap + inlet.pitch *( zlap + 4*nz_d) );
		}
	}
}

/* ================================= */

__global__ void do_u2d_upper_kernel(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField upper , cudaField dist , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	if(x < job.end.x && y < job.end.y){
		unsigned int xlap = x-LAP;
		unsigned int ylap = y-LAP;
		unsigned int ztmp = nz_lap_d - 1;
		get_Field_LAP(d , x , y , ztmp) = *(upper.ptr + xlap + upper.pitch *  ylap );
		get_Field_LAP(u , x , y , ztmp) = *(upper.ptr + xlap + upper.pitch *( ylap + 1*ny_d) ) + *(dist.ptr + xlap + dist.pitch * ylap );
		get_Field_LAP(v , x , y , ztmp) = *(upper.ptr + xlap + upper.pitch *( ylap + 2*ny_d) );
		get_Field_LAP(w , x , y , ztmp) = *(upper.ptr + xlap + upper.pitch *( ylap + 3*ny_d) );
		get_Field_LAP(T , x , y , ztmp) = *(upper.ptr + xlap + upper.pitch *( ylap + 4*ny_d) );
	}
}

__global__ void do_u_dist_upper_kernel(REAL sin_aoa , REAL cos_aoa , cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField dist , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	if(x < job.end.x && y < job.end.y){
		unsigned int xlap = x-LAP;
		unsigned int ylap = y-LAP;
		unsigned int ztmp = nz_lap_d - 1;
		get_Field_LAP(d , x , y , ztmp) = 1.0;
		get_Field_LAP(u , x , y , ztmp) =  cos_aoa + *(dist.ptr + xlap + dist.pitch * ylap );
		get_Field_LAP(v , x , y , ztmp) = 0.0;
		get_Field_LAP(w , x , y , ztmp) = sin_aoa;
		get_Field_LAP(T , x , y , ztmp) = 1.0;
	}
}

/* ============================================= */
__global__ void do_symmetry_kernel_m(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	if(x < job.end.x && y < job.end.y && z < job.end.z){
		unsigned int ys = 2*LAP - y;

		get_Field_LAP(d, x, y, z) = get_Field_LAP(d, x, ys, z); 
		get_Field_LAP(u, x, y, z) = get_Field_LAP(u, x, ys, z); 
		get_Field_LAP(v, x, y, z) = -1.0*get_Field_LAP(v, x, ys, z); 
		get_Field_LAP(w, x, y, z) = get_Field_LAP(w, x, ys, z); 
		get_Field_LAP(T, x, y, z) = get_Field_LAP(T, x, ys, z); 
	}
}

__global__ void do_symmetry_kernel_p(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	if(x < job.end.x && y < job.end.y && z < job.end.z){
		unsigned int ys = 2*(ny_lap_d - 1) - y;

		get_Field_LAP(d, x, y, z) = get_Field_LAP(d, x, ys, z); 
		get_Field_LAP(u, x, y, z) = get_Field_LAP(u, x, ys, z); 
		get_Field_LAP(v, x, y, z) = -1.0*get_Field_LAP(v, x, ys, z); 
		get_Field_LAP(w, x, y, z) = get_Field_LAP(w, x, ys, z); 
		get_Field_LAP(T, x, y, z) = get_Field_LAP(T, x, ys, z); 
	}
}

/* =============================================== */
__global__ void do_wall_kernel_T_V(REAL tw , cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField coeff , REAL HT , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	if(x < job.end.x && y < job.end.y){
		{
			unsigned int xlap = x-LAP;
			unsigned int ylap = y-LAP;
			get_Field_LAP(u , x , y , LAP) = *(coeff.ptr + xlap + coeff.pitch *  ylap ) * HT;
			get_Field_LAP(v , x , y , LAP) = *(coeff.ptr + xlap + coeff.pitch *( ylap + 1*ny_d) ) * HT;
			get_Field_LAP(w , x , y , LAP) = *(coeff.ptr + xlap + coeff.pitch *( ylap + 2*ny_d) ) * HT;
		}
		get_Field_LAP(T , x , y , LAP) = tw;
		get_Field_LAP(d , x , y , LAP) = (4.0 * get_Field_LAP(d , x,y,LAP+1) * get_Field_LAP(T , x , y , LAP+1) -  get_Field_LAP(d , x,y,LAP+2) * get_Field_LAP(T , x , y , LAP+2))/(3.0*tw);

	}
}
__global__ void do_wall_kernel_NT_V(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField coeff , REAL HT , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	if(x < job.end.x && y < job.end.y){
		{
			unsigned int xlap = x-LAP;
			unsigned int ylap = y-LAP;
			get_Field_LAP(u , x , y , LAP) = *(coeff.ptr + xlap + coeff.pitch *  ylap ) * HT;
			get_Field_LAP(v , x , y , LAP) = *(coeff.ptr + xlap + coeff.pitch *( ylap + 1*ny_d) ) * HT;
			get_Field_LAP(w , x , y , LAP) = *(coeff.ptr + xlap + coeff.pitch *( ylap + 2*ny_d) ) * HT;
		}
		get_Field_LAP(T , x , y , LAP) = (4.0 * get_Field_LAP(T , x , y , LAP+1) - get_Field_LAP(T , x , y , LAP+2))/3.0;
		get_Field_LAP(d , x , y , LAP) = (4.0 * get_Field_LAP(d , x , y , LAP+1) - get_Field_LAP(d , x , y , LAP+2))/3.0;

	}
}
__global__ void do_wall_kernel_T_NV(REAL tw , cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField coeff , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	if(x < job.end.x && y < job.end.y){
		
		get_Field_LAP(u , x , y , LAP) = 0.0;
		get_Field_LAP(v , x , y , LAP) = 0.0;
		get_Field_LAP(w , x , y , LAP) = 0.0;

		get_Field_LAP(T , x , y , LAP) = tw;
		get_Field_LAP(d , x , y , LAP) = (4.0 * get_Field_LAP(d , x,y,LAP+1) * get_Field_LAP(T , x , y , LAP+1) -  get_Field_LAP(d , x,y,LAP+2) * get_Field_LAP(T , x , y , LAP+2))/(3.0*tw);

	}
}

__global__ void do_wall_kernel_NT_NV(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField coeff , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	if(x < job.end.x && y < job.end.y){

		get_Field_LAP(u , x , y , LAP) = 0.0;
		get_Field_LAP(v , x , y , LAP) = 0.0;
		get_Field_LAP(w , x , y , LAP) = 0.0;

		get_Field_LAP(T , x , y , LAP) = (4.0 * get_Field_LAP(T , x , y , LAP+1) - get_Field_LAP(T , x , y , LAP+2))/3.0;
		get_Field_LAP(d , x , y , LAP) = (4.0 * get_Field_LAP(d , x , y , LAP+1) - get_Field_LAP(d , x , y , LAP+2))/3.0;

	}
}

/*  ---------------------------------------------- */
/* ======================================== */
void bc_user_Liftbody3d(){
	//-------------- boundary condition at i=1  (inlet) -----------------------------------------
	if (npx == 0)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , 1 , BlockDimY , BlockDimZ , 1 , ny , nz);

		cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(LAP+1, ny_lap ,nz_lap) );

		CUDA_LAUNCH(( do_u2d_inlet_kernel<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pu2d_inlet_d , job) ));
	}

	//---------------------bounrary at k=nz (upper) ------------------------------------------
	if (npz == NPZ0 - 1)
	{
		if (IFLAG_UPPERBOUNDARY == 0)
		{								 // Out of blow shock
			dim3 blockdim , griddim;
			cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);

			cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );

			CUDA_LAUNCH(( do_u_dist_upper_kernel<<<griddim , blockdim>>>( Sin_AOA , Cos_AOA ,*pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pu_dist_upper_d, job) ));
		}
		else if (IFLAG_UPPERBOUNDARY == 1)
		{ // In the blow shock
			dim3 blockdim , griddim;
			cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);

			cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );

			CUDA_LAUNCH(( do_u2d_upper_kernel<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pu2d_upper_d , *pu_dist_upper_d, job) ));
		}
	}

	//---------------------wall ------------------------------------

    REAL ht = 0.;

    if(BETA > 0.){
        for(int m = 0; m < MTMAX; m++){
            //ht = ht + TM[m] * sin((m + 1)*BETA*tt + 2.*PI*fait[m]);
            ht = ht + TM[m] * sin((m + 1)*BETA*tt);
        }
    }else{
        ht = 1.;
    }

	if (npz == 0)
	{
		if(v_dist_need){
			if(TW_postive){
				dim3 blockdim , griddim;
				cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);
		
				cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );
		
				CUDA_LAUNCH(( do_wall_kernel_T_V<<<griddim , blockdim>>>( TW , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pv_dist_coeff_d , ht , job) ));
			}else{
				dim3 blockdim , griddim;
				cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);
		
				cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );
		
				CUDA_LAUNCH(( do_wall_kernel_NT_V<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pv_dist_coeff_d, ht , job) ));
			}
		}else{
			if(TW_postive){
				dim3 blockdim , griddim;
				cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);
		
				cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );
		
				CUDA_LAUNCH(( do_wall_kernel_T_NV<<<griddim , blockdim>>>(TW , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pv_dist_coeff_d , job) ));
			}else{
				dim3 blockdim , griddim;
				cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);
		
				cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );
		
				CUDA_LAUNCH(( do_wall_kernel_NT_NV<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , *pv_dist_coeff_d , job) ));
			}
		}
	}
	//------------------------------------------------------------
	//------------ Symmetry -----------
	if (npy == 0 && IF_SYMMETRY == 1)
	{	
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz);

		cudaJobPackage job( dim3(LAP,0,LAP) , dim3(nx_lap, LAP ,nz_lap) );

		CUDA_LAUNCH(( do_symmetry_kernel_m<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));

	}

	if (npy == NPY0 - 1 && IF_SYMMETRY == 1)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz);

		cudaJobPackage job( dim3(LAP,ny_lap,LAP) , dim3(nx_lap, ny_2lap ,nz_lap) );

		CUDA_LAUNCH(( do_symmetry_kernel_p<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));
	}
}



/* =============================================================================== */

__global__ void simple_boundary_condition(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z ){
		
		get_Field_LAP(d , x , y , z) = 1.0;
		get_Field_LAP(u , x , y , z) = 1.0;
		get_Field_LAP(v , x , y , z) = 0.0;
		get_Field_LAP(w , x , y , z) = 0.0;
		get_Field_LAP(T , x , y , z) = 1.0;
	}
}

__global__ void out_boundary_condition(cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaJobPackage job){
	// with LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if(x < job.end.x && y < job.end.y && z < job.end.z ){
		
		get_Field_LAP(d , x , y , z) = get_Field_LAP(d , x-1 , y , z);
		get_Field_LAP(u , x , y , z) = get_Field_LAP(u , x-1 , y , z);
		get_Field_LAP(v , x , y , z) = get_Field_LAP(v , x-1 , y , z);
		get_Field_LAP(w , x , y , z) = get_Field_LAP(w , x-1 , y , z);
		get_Field_LAP(T , x , y , z) = get_Field_LAP(T , x-1 , y , z);
	}
}

void bc_user_Liftbody3d_simple(){
	//-------------- boundary condition at i=1  (inlet) -----------------------------------------
	if (npx == 0)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , 1 , BlockDimY , BlockDimZ , 1 , ny , nz);

		cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(LAP+1, ny_lap ,nz_lap) );

		CUDA_LAUNCH(( simple_boundary_condition<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));
	}

    if (npx == NPX0 - 1)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , 1 , BlockDimY , BlockDimZ , 1 , ny , nz);

		cudaJobPackage job( dim3(nx_lap,LAP,LAP) , dim3(nx_lap+1, ny_lap ,nz_lap) );

		CUDA_LAUNCH(( out_boundary_condition<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));
	}

	//---------------------bounrary at k=nz (upper) ------------------------------------------
	if (npz == NPZ0 - 1)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);

		cudaJobPackage job( dim3(LAP,LAP,nz_lap - 1) , dim3(nx_lap, ny_lap ,nz_lap) );

		CUDA_LAUNCH(( simple_boundary_condition<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));
	}

	//---------------------wall ------------------------------------



	if (npz == 0)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , 1 , nx , ny , 1);

		cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap ,LAP+1) );

		CUDA_LAUNCH(( simple_boundary_condition<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));
	}
	//------------------------------------------------------------
	//------------ Symmetry -----------
	if (npy == 0 && IF_SYMMETRY == 1)
	{	
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz);

		cudaJobPackage job( dim3(LAP,0,LAP) , dim3(nx_lap, LAP ,nz_lap) );

		CUDA_LAUNCH(( do_symmetry_kernel_m<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));

	}

	if (npy == NPY0 - 1 && IF_SYMMETRY == 1)
	{
		dim3 blockdim , griddim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz);

		cudaJobPackage job( dim3(LAP,ny_lap,LAP) , dim3(nx_lap, ny_2lap ,nz_lap) );

		CUDA_LAUNCH(( do_symmetry_kernel_p<<<griddim , blockdim>>>( *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job) ));
	}
}
#ifdef __cplusplus
}
#endif
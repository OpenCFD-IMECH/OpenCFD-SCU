//-------Boundary condition --------------------------------------------------------
#include "stdlib.h"
#include "stdio.h"
#include "parameters.h"
#include "utility.h"
#include "OCFD_boundary_Liftbody3D.h"
#include "OCFD_boundary_compression_conner.h"

#include "cuda_commen.h"
#include "cuda_utility.h"
#include "commen_kernel.h"
#include "parameters_d.h"

#ifdef __cplusplus
extern "C"{
#endif

void OCFD_bc()
{
	
	//---------------------------------------------------
    switch(IBC_USER){
        case 124:
		if(Init_stat == 0){
    		bc_user_Liftbody3d_simple();
    	}else{
    		bc_user_Liftbody3d();
    	}
		break;

		case 108:
		bc_user_Compression_conner();
		break;

		default:
		break;
	}

	//--------------------------------------------
	if (npx == 0)
	{	
		dim3 griddim , blockdim;
		cudaJobPackage job( dim3(0 , 0 , 0) , dim3(1 , ny , nz) );

		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , 1 , ny , nz);
		pri_to_cons_kernel<<<griddim , blockdim>>>(*pf_d , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job);
	}
	if (npx == NPX0 - 1)
	{
		dim3 griddim , blockdim;
		cudaJobPackage job( dim3(nx-1 , 0 , 0) , dim3(nx , ny , nz) );

		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , 1 , ny , nz);
		pri_to_cons_kernel<<<griddim , blockdim>>>(*pf_d , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job);
	}
	//------------------------------
	if (npy == 0)
	{
		dim3 griddim , blockdim;
		cudaJobPackage job( dim3(0 , 0 , 0) , dim3(nx , 1 , nz) );

		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , 1 , nz);
		pri_to_cons_kernel<<<griddim , blockdim>>>(*pf_d , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job);
	}

	if (npy == NPY0 - 1)
	{
		dim3 griddim , blockdim;
		cudaJobPackage job( dim3(0 , ny-1 , 0) , dim3(nx , ny , nz) );

		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , 1 , nz);
		pri_to_cons_kernel<<<griddim , blockdim>>>(*pf_d , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job);
	}
	//--------------------------
	if (npz == 0)
	{
		dim3 griddim , blockdim;
		cudaJobPackage job( dim3(0 , 0 , 0) , dim3(nx , ny , 1) );

		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , 1);
		pri_to_cons_kernel<<<griddim , blockdim>>>(*pf_d , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job);
	}

	if (npz == NPZ0 - 1)
	{
		dim3 griddim , blockdim;
		cudaJobPackage job( dim3(0 , 0 , nz-1) , dim3(nx , ny , nz) );

		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , 1);
		pri_to_cons_kernel<<<griddim , blockdim>>>(*pf_d , *pd_d , *pu_d , *pv_d , *pw_d , *pT_d , job);
	}
}


#ifdef __cplusplus
}
#endif

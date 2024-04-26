#include <math.h>

#include "OCFD_NS_Jacobian3d.h"
#include "parameters.h"
#include "OCFD_Schemes_Choose.h"
#include "OCFD_split.h"

#include "commen_kernel.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "OCFD_mpi_dev.h"
#include "parameters_d.h"
#include "OCFD_flux_charteric.h"

#ifdef __cplusplus
extern "C" {
#endif

void du_invis_Jacobian3d_init(cudaJobPackage job_in, cudaStream_t *stream){
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x+2*LAP, size.y+2*LAP, size.z+2*LAP);
	
	cudaJobPackage job( dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP), 
						dim3(job_in.end.x + LAP, job_in.end.y + LAP, job_in.end.z + LAP) );
						
	CUDA_LAUNCH(( sound_speed_kernel<<<griddim , blockdim, 0, *stream>>>(*pT_d , *pcc_d , job) ));
}


void du_invis_Jacobian3d_x(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, cudaStream_t *stream){

	OCFD_dx1(*fp, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAkx_d, *pAky_d, *pAkz_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

	OCFD_dx2(*fm, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAkx_d, *pAky_d, *pAkz_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

}

void du_invis_Jacobian3d_y(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, cudaStream_t *stream){

	OCFD_dy1(*fp, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAix_d, *pAiy_d, *pAiz_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

	OCFD_dy2(*fm, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAix_d, *pAiy_d, *pAiz_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

}


void du_invis_Jacobian3d_z(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, cudaStream_t *stream){

	OCFD_dz1(*fp, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAsx_d, *pAsy_d, *pAsz_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

	OCFD_dz2(*fm, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAsx_d, *pAsy_d, *pAsz_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

}

void dspec_invis_Jacobian3d_x(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, cudaStream_t *stream){

	OCFD_dx1_spec(*fp, *pdspec_d, *pAjac_d, *pAkx_d, *pAky_d, *pAkz_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

	OCFD_dx2_spec(*fm, *pdspec_d, *pAjac_d, *pAkx_d, *pAky_d, *pAkz_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

}

void dspec_invis_Jacobian3d_y(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, cudaStream_t *stream){

	OCFD_dy1_spec(*fp, *pdspec_d, *pAjac_d, *pAix_d, *pAiy_d, *pAiz_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

	OCFD_dy2_spec(*fm, *pdspec_d, *pAjac_d, *pAix_d, *pAiy_d, *pAiz_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

}


void dspec_invis_Jacobian3d_z(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, cudaStream_t *stream){

	OCFD_dz1_spec(*fp, *pdspec_d, *pAjac_d, *pAsx_d, *pAsy_d, *pAsz_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

	OCFD_dz2_spec(*fm, *pdspec_d, *pAjac_d, *pAsx_d, *pAsy_d, *pAsz_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

}

// ========================================================

void du_viscous_Jacobian3d_init(cudaStream_t *stream){

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

    OCFD_dx0(*pu_d, *puk_d, job, BlockDim, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pv_d, *pvk_d, job, BlockDim, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pw_d, *pwk_d, job, BlockDim, stream, D0_bound[0], D0_bound[1]);
	OCFD_dx0(*pT_d, *pTk_d, job, BlockDim, stream, D0_bound[0], D0_bound[1]);
	
    OCFD_dy0(*pu_d, *pui_d, job, BlockDim, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pv_d, *pvi_d, job, BlockDim, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pw_d, *pwi_d, job, BlockDim, stream, D0_bound[2], D0_bound[3]);
	OCFD_dy0(*pT_d, *pTi_d, job, BlockDim, stream, D0_bound[2], D0_bound[3]);
	
    OCFD_dz0(*pu_d, *pus_d, job, BlockDim, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pv_d, *pvs_d, job, BlockDim, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pw_d, *pws_d, job, BlockDim, stream, D0_bound[4], D0_bound[5]);
	OCFD_dz0(*pT_d, *pTs_d, job, BlockDim, stream, D0_bound[4], D0_bound[5]);

}


__device__ void vis_flux_s_ker(
	vis_flux vf,

	REAL *Akx,
	REAL *Aix,
	REAL *Asx,
	REAL *Aky,
	REAL *Aiy,
	REAL *Asy,
	REAL *Akz,
	REAL *Aiz,
	REAL *Asz,

	REAL *Amu,

	REAL *s11,
	REAL *s12,
	REAL *s13,
	REAL *s22,
	REAL *s23,
	REAL *s33,

	int x,
	int y,
	int z
){
	REAL ux, vx, wx;
	REAL uy, vy, wy;
	REAL uz, vz, wz;
	REAL div;

	REAL uk = get_Field(vf.uk, x-LAP, y-LAP, z-LAP);
	REAL ui = get_Field(vf.ui, x-LAP, y-LAP, z-LAP);
	REAL us = get_Field(vf.us, x-LAP, y-LAP, z-LAP);
	REAL vk = get_Field(vf.vk, x-LAP, y-LAP, z-LAP);
	REAL vi = get_Field(vf.vi, x-LAP, y-LAP, z-LAP);
	REAL vs = get_Field(vf.vs, x-LAP, y-LAP, z-LAP);
	REAL wk = get_Field(vf.wk, x-LAP, y-LAP, z-LAP);
	REAL wi = get_Field(vf.wi, x-LAP, y-LAP, z-LAP);
	REAL ws = get_Field(vf.ws, x-LAP, y-LAP, z-LAP);


	ux=uk* *Akx + ui* *Aix + us* *Asx;
	vx=vk* *Akx + vi* *Aix + vs* *Asx;
	wx=wk* *Akx + wi* *Aix + ws* *Asx;

	uy=uk* *Aky + ui* *Aiy + us* *Asy;
	vy=vk* *Aky + vi* *Aiy + vs* *Asy;
	wy=wk* *Aky + wi* *Aiy + ws* *Asy;
		
	uz=uk* *Akz + ui* *Aiz + us* *Asz;
	vz=vk* *Akz + vi* *Aiz + vs* *Asz;
	wz=wk* *Akz + wi* *Aiz + ws* *Asz;

	div=ux+vy+wz;
			
	*s11 = (2.0*ux-2.0/3.0*div) * *Amu;
	*s22 = (2.0*vy-2.0/3.0*div) * *Amu;
	*s33 = (2.0*wz-2.0/3.0*div) * *Amu;

	*s12 = (uy+vx)* *Amu;
	*s13 = (uz+wx)* *Amu;
	*s23 = (vz+wy)* *Amu;
}


__device__ void vis_flux_e_ker(
	vis_flux vf,

	REAL *Amu,
	REAL *Akx,
	REAL *Aky,
	REAL *Akz,
	REAL *Aix,
	REAL *Aiy,
	REAL *Aiz,
	REAL *Asx,
	REAL *Asy,
	REAL *Asz,

	REAL *s11,
	REAL *s12,
	REAL *s13,
	REAL *s22,
	REAL *s23,
	REAL *s33,

	REAL *E1,
	REAL *E2,
	REAL *E3,

	int x,
	int y,
	int z
){
	REAL Tx;
	REAL Ty;
	REAL Tz;
	REAL Amuk;

	REAL Tk = get_Field(vf.Tk, x-LAP, y-LAP, z-LAP);
	REAL Ti = get_Field(vf.Ti, x-LAP, y-LAP, z-LAP);
	REAL Ts = get_Field(vf.Ts, x-LAP, y-LAP, z-LAP);
	REAL u  = get_Field_LAP(vf.u, x, y, z);
	REAL v  = get_Field_LAP(vf.v, x, y, z);
	REAL w  = get_Field_LAP(vf.w, x, y, z);

	Amuk=*Amu * vis_flux_init_c_d;
			
	Tx=Tk* *Akx + Ti* *Aix + Ts* *Asx;	
	Ty=Tk* *Aky + Ti* *Aiy + Ts* *Asy;	
	Tz=Tk* *Akz + Ti* *Aiz + Ts* *Asz;

	*E1=u* *s11 + v* *s12 + w* *s13 + Amuk*Tx;
	*E2=u* *s12 + v* *s22 + w* *s23 + Amuk*Ty;
	*E3=u* *s13 + v* *s23 + w* *s33 + Amuk*Tz;
}


__device__ void vis_flus_ev_ker(
	vis_flux vf,

	REAL *s11,
	REAL *s12,
	REAL *s13,
	REAL *s22,
	REAL *s23,
	REAL *s33,

	REAL *E1,
	REAL *E2,
	REAL *E3,

	cudaField Ev1,
	cudaField Ev2,
	cudaField Ev3,
	cudaField Ev4,

	int x,
	int y,
	int z
){
	REAL akx , aky , akz;
	{
		REAL Aj1;
		Aj1 = get_Field_LAP(vf.Ajac , x,y,z);

		akx = get_Field_LAP(vf.Ax, x, y, z)*Aj1;
		aky = get_Field_LAP(vf.Ay, x, y, z)*Aj1;
		akz = get_Field_LAP(vf.Az, x, y, z)*Aj1;
	}
	
	get_Field_LAP(Ev1, x, y, z) = ( akx* *s11 + aky* *s12 + akz* *s13 );
	get_Field_LAP(Ev2, x, y, z) = ( akx* *s12 + aky* *s22 + akz* *s23 ); 
	get_Field_LAP(Ev3, x, y, z) = ( akx* *s13 + aky* *s23 + akz* *s33 );
	get_Field_LAP(Ev4, x, y, z) = ( akx* *E1  + aky* *E2  + akz* *E3  );
}


__global__ void vis_flux_ker(

	vis_flux vf,

	cudaField Ev1,
	cudaField Ev2,
	cudaField Ev3,
	cudaField Ev4,

	cudaJobPackage job)
{
	// eyes on cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;


    if( x<job.end.x && y<job.end.y && z<job.end.z){

		REAL s11, s12, s13, s22, s23, s33;
		REAL E1, E2, E3;
	
		REAL Akx = get_Field_LAP(vf.Akx, x, y, z);
		REAL Aix = get_Field_LAP(vf.Aix, x, y, z);
		REAL Asx = get_Field_LAP(vf.Asx, x, y, z);
		REAL Aky = get_Field_LAP(vf.Aky, x, y, z);
		REAL Aiy = get_Field_LAP(vf.Aiy, x, y, z);
		REAL Asy = get_Field_LAP(vf.Asy, x, y, z);
		REAL Akz = get_Field_LAP(vf.Akz, x, y, z);
		REAL Aiz = get_Field_LAP(vf.Aiz, x, y, z);
		REAL Asz = get_Field_LAP(vf.Asz, x, y, z);
	
		REAL Amu = get_Field(vf.Amu, x-LAP, y-LAP, z-LAP);

		vis_flux_s_ker(vf,&Akx,&Aix,&Asx,&Aky,&Aiy,&Asy,&Akz,&Aiz,&Asz,&Amu,&s11,&s12,&s13,&s22,&s23,&s33,x,y,z);

		vis_flux_e_ker(vf,&Amu,&Akx,&Aky,&Akz,&Aix,&Aiy,&Aiz,&Asx,&Asy,&Asz,
			&s11,&s12,&s13,&s22,&s23,&s33,&E1,&E2,&E3,x,y,z);

		vis_flus_ev_ker(vf,&s11,&s12,&s13,&s22,&s23,&s33,&E1,&E2,&E3,
			Ev1,Ev2,Ev3,Ev4,x,y,z);
	}
}


void du_viscous_Jacobian3d_x_init(cudaStream_t *stream){

	dim3 blockdim , griddim;

    uint32_t BlockDimX1 = 8;
    uint32_t BlockDimY1 = 4;
    uint32_t BlockDimZ1 = 4;
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX1, BlockDimY1, BlockDimZ1, nx, ny, nz);

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

	vis_flux vis_flux_parameter = {*puk_d,*pvk_d,*pwk_d,*pui_d,*pvi_d,*pwi_d,*pus_d,*pvs_d,*pws_d,
				                   *pTk_d,*pTi_d,*pTs_d,*pAmu_d,
		                           *pu_d,*pv_d,*pw_d,*pAkx_d,*pAky_d,*pAkz_d,
		                           *pAjac_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d};

	CUDA_LAUNCH(( vis_flux_ker<<<griddim , blockdim, 0, *stream>>>(vis_flux_parameter, *pEv1_d, *pEv2_d, *pEv3_d, *pEv4_d, job) ));

}

void du_viscous_Jacobian3d_x_final(cudaJobPackage job_in, cudaStream_t *stream){

	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    OCFD_dx0(*pEv1_d, *vis_u_d, job_in, BlockDim, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pEv2_d, *vis_v_d, job_in, BlockDim, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pEv3_d, *vis_w_d, job_in, BlockDim, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pEv4_d, *vis_T_d, job_in, BlockDim, stream, D0_bound[0], D0_bound[1]);

	cudaJobPackage job(dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP) , 
	                   dim3(job_in.end.x - LAP, job_in.end.y - LAP, job_in.end.z - LAP));

	int size_du = pdu_d->pitch*ny*nz;
	cudaField tmp_du;
	tmp_du.pitch = pdu_d->pitch;

	tmp_du.ptr = pdu_d->ptr + size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_u_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_v_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_w_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_T_d, *pAjac_d, job) ));
}

void du_viscous_Jacobian3d_y_init(cudaStream_t *stream){

	dim3 blockdim , griddim;

    uint32_t BlockDimX1 = 8;
    uint32_t BlockDimY1 = 4;
    uint32_t BlockDimZ1 = 4;
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX1, BlockDimY1, BlockDimZ1, nx, ny, nz);

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

	vis_flux vis_flux_parameter = {*puk_d,*pvk_d,*pwk_d,*pui_d,*pvi_d,*pwi_d,*pus_d,*pvs_d,*pws_d,
								   *pTk_d,*pTi_d,*pTs_d,*pAmu_d,
								   *pu_d,*pv_d,*pw_d,*pAix_d,*pAiy_d,*pAiz_d,
								   *pAjac_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d};

    CUDA_LAUNCH(( vis_flux_ker<<<griddim , blockdim, 0, *stream>>>(vis_flux_parameter, *pEv1_d, *pEv2_d, *pEv3_d, *pEv4_d, job) ));

}

void du_viscous_Jacobian3d_y_final(cudaJobPackage job_in, cudaStream_t *stream){

	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    OCFD_dy0(*pEv1_d, *vis_u_d, job_in, BlockDim, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pEv2_d, *vis_v_d, job_in, BlockDim, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pEv3_d, *vis_w_d, job_in, BlockDim, stream, D0_bound[2], D0_bound[3]);
	OCFD_dy0(*pEv4_d, *vis_T_d, job_in, BlockDim, stream, D0_bound[2], D0_bound[3]);

	cudaJobPackage job(dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP) , 
					   dim3(job_in.end.x - LAP, job_in.end.y - LAP, job_in.end.z - LAP));
					   
	int size_du = pdu_d->pitch*ny*nz;
	cudaField tmp_du;
	tmp_du.pitch = pdu_d->pitch;

	tmp_du.ptr = pdu_d->ptr + size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_u_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_v_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_w_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_T_d, *pAjac_d, job) ));
}


void du_viscous_Jacobian3d_z_init(cudaStream_t *stream){

	dim3 blockdim , griddim;

    uint32_t BlockDimX1 = 8;
    uint32_t BlockDimY1 = 4;
    uint32_t BlockDimZ1 = 4;
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX1, BlockDimY1, BlockDimZ1, nx, ny, nz);

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

	vis_flux vis_flux_parameter = {*puk_d,*pvk_d,*pwk_d,*pui_d,*pvi_d,*pwi_d,*pus_d,*pvs_d,*pws_d,
								   *pTk_d,*pTi_d,*pTs_d,*pAmu_d,
								   *pu_d,*pv_d,*pw_d,*pAsx_d,*pAsy_d,*pAsz_d,
								   *pAjac_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d};

    CUDA_LAUNCH(( vis_flux_ker<<<griddim , blockdim, 0, *stream>>>(vis_flux_parameter, *pEv1_d, *pEv2_d, *pEv3_d, *pEv4_d, job) ));

}


void du_viscous_Jacobian3d_z_final(cudaJobPackage job_in, cudaStream_t *stream){

	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

	OCFD_dz0(*pEv1_d, *vis_u_d, job_in, BlockDim, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pEv2_d, *vis_v_d, job_in, BlockDim, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pEv3_d, *vis_w_d, job_in, BlockDim, stream, D0_bound[4], D0_bound[5]);
	OCFD_dz0(*pEv4_d, *vis_T_d, job_in, BlockDim, stream, D0_bound[4], D0_bound[5]);

	cudaJobPackage job(dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP) , 
					   dim3(job_in.end.x - LAP, job_in.end.y - LAP, job_in.end.z - LAP));
					   
	int size_du = pdu_d->pitch*ny*nz;
	cudaField tmp_du;
	tmp_du.pitch = pdu_d->pitch;

	tmp_du.ptr = pdu_d->ptr + size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_u_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_v_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_w_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_T_d, *pAjac_d, job) ));
}

__global__ void boundary_symmetry_pole_vis_y_ker_m(
	cudaField Ev1,
	cudaField Ev2,
	cudaField Ev3,
	cudaField Ev4,
	cudaJobPackage job){

	// eyes on Bottom holo cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
		unsigned int y1 = 2*LAP - y;

		get_Field_LAP(Ev1 , x,y,z) = - get_Field_LAP(Ev1 , x,y1,z);
		get_Field_LAP(Ev2 , x,y,z) =   get_Field_LAP(Ev2 , x,y1,z);
		get_Field_LAP(Ev3 , x,y,z) = - get_Field_LAP(Ev3 , x,y1,z);
		get_Field_LAP(Ev4 , x,y,z) = - get_Field_LAP(Ev4 , x,y1,z);
	}
}

__global__ void boundary_symmetry_pole_vis_y_ker_p(
	cudaField Ev1,
	cudaField Ev2,
	cudaField Ev3,
	cudaField Ev4,
	cudaJobPackage job){

	// eyes on Top holo cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
		unsigned int y1 = 2*(ny_d+LAP-1) - y;

		get_Field_LAP(Ev1 , x,y,z) = - get_Field_LAP(Ev1 , x,y1,z);
		get_Field_LAP(Ev2 , x,y,z) =   get_Field_LAP(Ev2 , x,y1,z);
		get_Field_LAP(Ev3 , x,y,z) = - get_Field_LAP(Ev3 , x,y1,z);
		get_Field_LAP(Ev4 , x,y,z) = - get_Field_LAP(Ev4 , x,y1,z);
	}
}

void boundary_symmetry_pole_vis_y(cudaStream_t *stream){
	dim3 blockdim , griddim;
//    symmetry or pole boundary condition for viscous term
    if(IF_SYMMETRY == 1){
        if(npy == 0){
		    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , LAP , BlockDimZ , nx , LAP , nz);
		    cudaJobPackage job(dim3(LAP , 0 , LAP) , dim3(nx_lap , LAP , nz_lap));
		    CUDA_LAUNCH(( boundary_symmetry_pole_vis_y_ker_m<<<griddim , blockdim, 0, *stream>>>(*pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d , job) ));
	    }
	    if(npy == NPY0-1){
		    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , LAP , BlockDimZ , nx , LAP , nz);
		    cudaJobPackage job(dim3(LAP , ny_lap , LAP) , dim3(nx_lap , ny_2lap , nz_lap));
		    CUDA_LAUNCH(( boundary_symmetry_pole_vis_y_ker_p<<<griddim , blockdim, 0, *stream>>>(*pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d , job) ));
    	}
    }
}

#ifdef __cplusplus
}
#endif

#include "hip/hip_runtime.h"
#ifndef __OCFD_NS_JACOBIAN_H
#define __OCFD_NS_JACOBIAN_H
#include "parameters.h"
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

void du_invis_Jacobian3d_init(cudaJobPackage job_in, hipStream_t *stream);

void du_invis_Jacobian3d_x(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream);
//void du_invis_Jacobian3d_x(cudaJobPackage job_in, cudaSoA *fp_x, cudaSoA *fm_x, cudaSoA *fp_y, cudaSoA *fm_y, cudaSoA *fp_z, cudaSoA *fm_z, hipStream_t *stream);
void du_invis_Jacobian3d_y(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream);
void du_invis_Jacobian3d_z(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream);


void du_viscous_Jacobian3d_init(hipStream_t *stream);

void du_viscous_Jacobian3d_x_init(hipStream_t *stream);
void du_viscous_Jacobian3d_x_final(cudaJobPackage job_in, hipStream_t *stream);
void du_viscous_Jacobian3d_y_init(hipStream_t *stream);
void du_viscous_Jacobian3d_y_final(cudaJobPackage job_in, hipStream_t *stream);
void du_viscous_Jacobian3d_z_init(hipStream_t *stream);
void du_viscous_Jacobian3d_z_final(cudaJobPackage job_in, hipStream_t *stream);

void boundary_symmetry_pole_vis_y(hipStream_t *stream);


/* ============= */
void vis_flux_ker_x_warp();
void vis_flux_ker_y_warp();
void vis_flux_ker_z_warp();
void boundary_symmetry_pole_vis_y_warp();


typedef struct vis_flux_
{
    cudaField uk;
	cudaField vk;
	cudaField wk;
	cudaField ui;
	cudaField vi;
	cudaField wi;
	cudaField us;
	cudaField vs;
	cudaField ws;

	cudaField Tk;
	cudaField Ti;
	cudaField Ts;

	cudaField Amu;

	cudaField u;
	cudaField v;
	cudaField w;

	cudaField Ax;
	cudaField Ay;
	cudaField Az;

	cudaField Ajac;
	cudaField Akx;
	cudaField Aky;
	cudaField Akz;
	cudaField Aix;
	cudaField Aiy;
	cudaField Aiz;
	cudaField Asx;
	cudaField Asy;
	cudaField Asz;

	cudaField Ev1;
	cudaField Ev2;
	cudaField Ev3;
	cudaField Ev4;
} vis_flux;


/* ===================== */
/*#if ((defined __NVCC__) || (defined __HIPCC__))
__global__ void vis_flux_ker(
cudaField Akx,cudaField Aky,cudaField Akz,cudaField Ajac,
cudaField s11,cudaField s12,cudaField s13,cudaField s22,cudaField s23,cudaField s33,cudaField E1,cudaField E2,cudaField E3,
cudaField Ev1,cudaField Ev2,cudaField Ev3,cudaField Ev4,
cudaJobPackage job);

__global__ void vis_flux_init_ker(
cudaField uk,cudaField vk,cudaField wk,cudaField Tk,cudaField ui,cudaField vi,cudaField wi,cudaField Ti,cudaField us,cudaField vs,cudaField ws,cudaField Ts,
cudaField Akx,cudaField Aky,cudaField Akz,cudaField Aix,cudaField Aiy,cudaField Aiz,cudaField Asx,cudaField Asy,cudaField Asz,
cudaField Amu,
cudaField u,cudaField v,cudaField w,
cudaField s11,cudaField s22,cudaField s33,cudaField s12,cudaField s13,cudaField s23,
cudaField E1,cudaField E2,cudaField E3,
cudaJobPackage job
);

__global__ void final_flux_ker(cudaSoA du , cudaField df1 , cudaField df2 , cudaField df3 , cudaField df4 , cudaField AJac, cudaJobPackage job);
#endif
*/
#ifdef __cplusplus
}
#endif
#endif

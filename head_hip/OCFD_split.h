#include "hip/hip_runtime.h"
#ifndef __OCFD_SPLIT_H
#define __OCFD_SPLIT_H
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif
//__global__ void split_Jac3d_Stager_Warming_ker(cudaField d0, cudaField u0, cudaField v0, cudaField w0, cudaField cc0, cudaSoA fp, cudaSoA fm, cudaField Akx, cudaField Aky, cudaField Akz, REAL tmp0, REAL split_C1, REAL split_C3, cudaJobPackage job);

typedef struct sw_split_
{
	cudaField d;
	cudaField u;
	cudaField v;
	cudaField w;
	cudaField cc;

	cudaField Akx;
	cudaField Aky;
	cudaField Akz;
	cudaField Aix;
	cudaField Aiy;
	cudaField Aiz;
	cudaField Asx;
	cudaField Asy;
	cudaField Asz;
} sw_split;

__global__ void split_Jac3d_Stager_Warming_ker(sw_split sw, cudaSoA fp_x, cudaSoA fm_x, cudaSoA fp_y, cudaSoA fm_y, cudaSoA fp_z, cudaSoA fm_z, REAL tmp0, REAL split_C1, REAL split_C3, cudaJobPackage job);

void Stager_Warming(cudaJobPackage job_in, cudaSoA *fp_x, cudaSoA *fm_x, cudaSoA *fp_y, cudaSoA *fm_y, cudaSoA *fp_z, cudaSoA *fm_z, hipStream_t *stream);

typedef struct sw_split_out_
{
	cudaField d;
	cudaField u;
	cudaField v;
	cudaField w;
	cudaField cc;

	cudaField Ax;
	cudaField Ay;
	cudaField Az;
} sw_split_out;

void Stager_Warming_out(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream);

#ifdef __cplusplus
}
#endif
#endif
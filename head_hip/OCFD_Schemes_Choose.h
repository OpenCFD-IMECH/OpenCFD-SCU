#ifndef __OCFD_SCHEME_CHOOSE_H
#define __OCFD_SCHEME_CHOOSE_H

#include "parameters.h"
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

void OCFD_dx0(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);
void OCFD_dx1(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);
void OCFD_dx2(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);

void OCFD_dy0(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);
void OCFD_dy1(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);
void OCFD_dy2(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);

void OCFD_dz0(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);
void OCFD_dz1(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);
void OCFD_dz2(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, hipStream_t *stream, int boundl, int boundr);

void OCFD_dx0_jac(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream, int bound);
void OCFD_dy0_jac(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream, int bound);
void OCFD_dz0_jac(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream, int bound);
#ifdef __cplusplus
}
#endif

#endif
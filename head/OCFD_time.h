#ifndef __OCFD_TIME_H
#define __OCFD_TIME_H

#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

void OCFD_time_advance(int KRK);
void OCFD_time_advance_plus(int KRK);

// This defines a RHS source to the N-S equation 
// __global__ void source_kernel(cudaSoA du , cudaField rho , cudaField v, cudaField w, cudaField yy, cudaField zz, cudaJobPackage job);
__global__ void OCFD_time_advance_ker1(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job);
__global__ void OCFD_time_advance_ker2(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job);
__global__ void OCFD_time_advance_ker3(cudaSoA f , cudaSoA fn , cudaSoA du , cudaSoA pf_lap , cudaJobPackage job);

__global__ void OCFD_spec_time_advance_ker1(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job, int n);
__global__ void OCFD_spec_time_advance_ker2(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job, int n);
__global__ void OCFD_spec_time_advance_ker3(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job, int n);

#ifdef __cplusplus
}
#endif

#endif
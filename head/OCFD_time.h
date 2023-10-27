#ifndef __OCFD_TIME_H
#define __OCFD_TIME_H

#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

void OCFD_time_advance(int KRK);
void OCFD_time_advance_plus(int KRK);

// __global__ void source_kernel(cudaSoA du , cudaField rho , cudaField v, cudaField w, cudaField yy, cudaField zz, cudaJobPackage job);
__global__ void OCFD_time_advance_ker1(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job);
__global__ void OCFD_time_advance_ker2(cudaSoA f , cudaSoA fn , cudaSoA du , cudaJobPackage job);
__global__ void OCFD_time_advance_ker3(cudaSoA f , cudaSoA fn , cudaSoA du , cudaSoA pf_lap , cudaJobPackage job);

#ifdef __cplusplus
}
#endif

#endif
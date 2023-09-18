#include "hip/hip_runtime.h"
#ifndef __OCFD_FLUX_CHARTERIC_H
#define __OCFD_FLUX_CHARTERIC_H
#include "parameters.h"
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

__global__ void OCFD_weno7_SYMBO_character_P_kernel(int WENO_LMT_FLAG, 
dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField u, cudaField v,
cudaField w, cudaField cc, cudaField Ax, cudaField Ay, cudaField Az,
cudaJobPackage job);


__global__ void OCFD_weno7_SYMBO_character_M_kernel(int WENO_LMT_FLAG, 
dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField u, cudaField v,
cudaField w, cudaField cc, cudaField Ax, cudaField Ay, cudaField Az,
cudaJobPackage job);


__global__ void OCFD_HybridAuto_character_P_kernel( 
dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField u, cudaField v,
cudaField w, cudaField cc, cudaField Ax, cudaField Ay, cudaField Az,
cudaField_int scheme, cudaJobPackage job);


__global__ void OCFD_HybridAuto_character_M_kernel(
dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField u, cudaField v,
cudaField w, cudaField cc, cudaField Ax, cudaField Ay, cudaField Az,
cudaField_int scheme, cudaJobPackage job);


__global__ void OCFD_HybridAuto_character_P_Jameson_kernel( 
dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField u, cudaField v,
cudaField w, cudaField cc, cudaField Ax, cudaField Ay, cudaField Az,
cudaField_int scheme, cudaJobPackage job);


__global__ void OCFD_HybridAuto_character_M_Jameson_kernel(
dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField u, cudaField v,
cudaField w, cudaField cc, cudaField Ax, cudaField Ay, cudaField Az,
cudaField_int scheme, cudaJobPackage job);

#ifdef __cplusplus
}
#endif
#endif
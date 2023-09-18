#ifndef __OCFD_SCHEME_H
#define __OCFD_SCHEME_H

#include "parameters.h"
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif
    
__device__ REAL minmod2(REAL x1, REAL x2);
__device__ REAL minmod4(REAL x1, REAL x2, REAL x3, REAL x4);

__global__ void OCFD_CD6_kernel(dim3 flagxyzb, cudaField pf, cudaField pfy, cudaJobPackage job);
__global__ void OCFD_CD8_kernel(dim3 flagxyzb, cudaField pf, cudaField pfy, cudaJobPackage job);

__global__ void OCFD_dx0_CD6_kernel(cudaField pf , cudaField pfx , cudaJobPackage job);
__global__ void OCFD_dy0_CD6_kernel(cudaField pf , cudaField pfy , cudaJobPackage job);
__global__ void OCFD_dz0_CD6_kernel(cudaField pf , cudaField pfz , cudaJobPackage job);

__global__ void OCFD_dx0_CD8_kernel(cudaField pf , cudaField pfx , cudaJobPackage job);
__global__ void OCFD_dy0_CD8_kernel(cudaField pf , cudaField pfy , cudaJobPackage job);
__global__ void OCFD_dz0_CD8_kernel(cudaField pf , cudaField pfz , cudaJobPackage job);


__device__ REAL OCFD_weno7_SYMBO_kernel_P(int WENO_LMT_FLAG, REAL *stencil);
__device__ REAL OCFD_weno7_SYMBO_kernel_M(int WENO_LMT_FLAG, REAL *stencil);
__global__ void OCFD_weno7_SYMBO_P_kernel(int i, int WENO_LMT_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);
__global__ void OCFD_weno7_SYMBO_M_kernel(int i, int WENO_LMT_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);


__device__ void aa(int a, ...);

__device__ REAL OCFD_weno5_kernel_P(REAL *stencil);
__device__ REAL OCFD_weno5_kernel_M(REAL *stencil);
__global__ void OCFD_weno5_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);
__global__ void OCFD_weno5_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);

__device__ REAL OCFD_weno7_kernel_P(REAL *stencil);
__device__ REAL OCFD_weno7_kernel_M(REAL *stencil);
__global__ void OCFD_weno7_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);
__global__ void OCFD_weno7_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);


__device__ REAL OCFD_NND2_kernel_P(REAL *stencil);
__device__ REAL OCFD_NND2_kernel_M(REAL *stencil);
__global__ void OCFD_NND2_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);
__global__ void OCFD_NND2_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);


__device__ REAL OCFD_UP7_kernel_P(REAL *stencil);
__device__ REAL OCFD_UP7_kernel_M(REAL *stencil);
__global__ void OCFD_UP7_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);
__global__ void OCFD_UP7_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);


__device__ REAL OCFD_OMP6_kernel_P(int OMP6_FLAG, REAL *stencil);
__device__ REAL OCFD_OMP6_kernel_M(int OMP6_FLAG, REAL *stencil);
__global__ void OCFD_OMP6_P_kernel(int i, int OMP6_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);
__global__ void OCFD_OMP6_M_kernel(int i, int OMP6_FLAG, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaJobPackage job);


__device__ REAL OCFD_weno5_kernel_P(REAL *stencil);
__device__ REAL OCFD_weno5_kernel_M(REAL *stencil);


__device__ int get_data_kernel(int flagxyz, dim3 *coords, cudaSoA f, int num, REAL *stencil, int ka1, int kb1, REAL *sort, cudaJobPackage job);
//__device__ void put_du_p_kernel(dim3 flagxyzb, dim3 coords, REAL tmp_r, REAL tmp_l, cudaSoA du, REAL *stencil, int num, cudaField Ajac, cudaJobPackage job);
__device__ void put_du_p_kernel(dim3 flagxyz, dim3 coords, REAL tmp_r, REAL tmp_l, cudaSoA du, int num, cudaField Ajac, cudaJobPackage job);
__device__ void put_du_m_kernel(dim3 flagxyz, dim3 coords, REAL tmp_r, REAL tmp_l, cudaSoA du, int num, cudaField Ajac, cudaJobPackage job);
//__device__ void put_du_m_kernel(dim3 flagxyzb, dim3 coords, REAL tmp_r, REAL tmp_l, cudaSoA du, REAL *stencil, int num, cudaField Ajac, cudaJobPackage job);
#ifdef __cplusplus
}
#endif

#endif
#ifndef __OCFD_SCHEMES_HYBRID_AUTO_H
#define __OCFD_SCHEMES_HYBRID_AUTO_H

#include "parameters.h"
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

typedef struct HybridAuto_TYPE_{ //used in SCHEME_HYBRIDAUTO
    REAL *P_intvs;
    cudaField_int *scheme_x;
    cudaField_int *scheme_y;
    cudaField_int *scheme_z;
    int Num_Patch_zones;
    int *zones; //[6][Patch_max]
    REAL *Pa_zones;
    int IF_Smooth_dp;
    int Style; 

} HybridAuto_TYPE;
    
void Set_Scheme_HybridAuto(cudaStream_t *stream);
void Comput_P(cudaField *d, cudaField *T, cudaField *P, cudaStream_t *stream);
void Comput_grad(cudaField *f, cudaStream_t *stream);
void Smoothing_dp(cudaStream_t *stream);
void Patch_zones(cudaStream_t *stream);
void Boundary_dp(cudaStream_t *stream);
void Comput_Scheme_point(cudaStream_t *stream);
void Comput_Scheme_point_Jameson(cudaStream_t *stream);

__global__ void add_kernel(REAL *g_odata, int g_odata_size);
__device__ REAL warpReduce(REAL mySum);

__device__ int get_Hyscheme_flag_p_kernel(int flagxyz, dim3 coords, cudaField_int scheme, cudaJobPackage job);
__device__ int get_Hyscheme_flag_m_kernel(int flagxyz, dim3 coords, cudaField_int scheme, cudaJobPackage job);

__global__ void OCFD_HybridAuto_P_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job);
__global__ void OCFD_HybridAuto_M_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job);

__global__ void OCFD_HybridAuto_P_Jameson_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job);
__global__ void OCFD_HybridAuto_M_Jameson_kernel(int i, dim3 flagxyzb, cudaSoA f, cudaSoA du, cudaField Ajac, cudaField_int scheme, cudaJobPackage job);

void HybridAuto_scheme_IO();
void HybridAuto_scheme_Proportion();
void modify_NT(cudaStream_t *stream);

#ifdef __cplusplus
}
#endif

#endif
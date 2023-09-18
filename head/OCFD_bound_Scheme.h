#ifndef __OCFD_BOUND_SCHEME_H
#define __OCFD_BOUND_SCHEME_H

#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif


void OCFD_Dx0_bound(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, cudaStream_t *stream);
void OCFD_Dy0_bound(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, cudaStream_t *stream);
void OCFD_Dz0_bound(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, cudaStream_t *stream);


void OCFD_bound(dim3 *flagxyzb, int boundp, int boundm, cudaJobPackage job);

void OCFD_bound_non_ref(dim3 *flagxyzb, int Non_ref, cudaJobPackage job);

__device__ int OCFD_D0bound_scheme_kernel(REAL* tmp, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, cudaJobPackage job);

__device__ int OCFD_bound_scheme_kernel_p(REAL* flag, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, int kb1, cudaJobPackage job);
__device__ int OCFD_bound_scheme_kernel_m(REAL* flag, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, int kb1, cudaJobPackage job);

#ifdef __cplusplus
}
#endif
#endif

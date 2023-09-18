#ifndef __OCFD_COMPUTE_JACOBIAN3D_H
#define __OCFD_COMPUTE_JACOBIAN3D_H
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif

void Init_Jacobian3d();
void Comput_Jacobian3d();
void comput_Jac3d();
void boundary_Jac3d_Axx();
void boundary_Jac3d_Liftbody_Ajac();

void boundary_Jac3d_kernal_y_ramp_wall(REAL seta);
void boundary_Jac3d_kernal_z_cone_wall(REAL seta1, REAL seta2);

#if ((defined __NVCC__) || (defined __HIPCC__))
#include "cuda_commen.h"
__global__ void comput_Jac3d_kernal(
    cudaField xi,cudaField xj,cudaField xk,cudaField yi,cudaField yj,cudaField yk,cudaField zi,cudaField zj,cudaField zk,cudaField Akx,
    cudaField Aky,cudaField Akz,cudaField Aix,cudaField Aiy,cudaField Aiz,cudaField Asx,cudaField Asy,cudaField Asz,cudaField Ajac,
    cudaJobPackage job);
#endif
#ifdef __cplusplus
}
#endif
#endif
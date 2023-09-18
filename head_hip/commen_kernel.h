#include "hip/hip_runtime.h"
#ifndef _COMMEN_KERNEL_H
#define _COMMEN_KERNEL_H
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

__global__ void cuda_mem_value_init(REAL value , REAL * ptr , unsigned int pitch , unsigned int size_x , unsigned int size_y , unsigned int size_z);
void cuda_mem_value_init_warp(REAL value , REAL * ptr , unsigned int pitch , unsigned int size_x , unsigned int size_y , unsigned int size_z);

// eyes on no-lap region
__global__ void pri_to_cons_kernel(cudaSoA pcons , cudaField pd , cudaField pu , cudaField pv , cudaField pw , cudaField pT , cudaJobPackage job);
void pri_to_cons_kernel_warp(cudaSoA *pcons , cudaField *pd , cudaField *pu , cudaField *pv , cudaField *pw , cudaField *pT , cudaJobPackage job_in , dim3 blockdim_in );

__global__ void cons_to_pri_kernel(cudaSoA f, cudaField d , cudaField u , cudaField v , cudaField w , cudaField T , cudaField P , cudaJobPackage job);
void get_duvwT();

__global__ void get_Amu_kernal(cudaField Amu , cudaField T , cudaJobPackage job);
void get_Amu();

__global__ void sound_speed_kernel(cudaField T , cudaField cc , cudaJobPackage job);


__global__ void YF_Pe_XF(cudaField yF , cudaField xF , cudaField AJac , cudaJobPackage job);
__global__ void ZF_e_XF_P_YF(cudaField out , cudaField xF , cudaField yF , cudaJobPackage job);
__global__ void ZF_e_XF_P_YF_LAP(cudaField out , cudaField xF , cudaField yF , cudaJobPackage job);
__global__ void ZF_Pe_XF_P_YF(cudaField zF , cudaField xF , cudaField yF , cudaField AJac , cudaJobPackage job);




/* ========================================= */
// inline function
// #include "config_parameters.h"
// #include "cuda_commen.h"
// #include "cuda_utility.h"
// #include "parameters_d.h"
// __device__ inline void cons_to_pri_dev_fun(cudaField & d , cudaField & u , cudaField & v , cudaField & w , cudaField & T , cudaField & P , REAL & f0 , REAL & f1 , REAL & f2 , REAL & f3 , REAL & f4 ){
//         get_Field_LAP(d , x+LAP , y+LAP , z+LAP) = f0;

//         REAL u = f1/f0;
//         get_Field_LAP(u , x+LAP , y+LAP , z+LAP) = u;

//         REAL v = f2/f0;
//         get_Field_LAP(v , x+LAP , y+LAP , z+LAP) = v;

//         REAL w = f3/f0;
//         get_Field_LAP(w , x+LAP , y+LAP , z+LAP) = w;

//         REAL tmp = f4 - 0.5*f0*(u*u + v*v + w*w);
//         get_Field_LAP(T , x+LAP , y+LAP , z+LAP) = tmp/(f0*Cv_d);
//         tmp = tmp/d1;
//         get_Field_LAP(P , x+LAP , y+LAP , z+LAP) = tmp*(Gamma_d - 1.0);
// }

// __device__ inline void get_Amu_kernal(cudaField & Amu , REAL & t){
//         get_Field(Amu , x,y,z) = amu_C0_d * sqrt(t * t * t) / (Tsb_d + t);
// }


// __device__ inline void sound_speed_kernel(cudaField & cc , REAL & t){
//         get_Field_LAP(cc , x,y,z) = sqrt( get_Field_LAP(T , x,y,z) )/Ama_d;
// }
#ifdef __cplusplus
}
#endif
#endif

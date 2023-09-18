#ifndef __OCFD_WARP_SHUFFLE_H
#define __OCFD_WARP_SHUFFLE_H
#include "cuda_commen.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#ifdef __cplusplus
extern "C"{
#endif


#ifdef __NVCC__
__device__ __forceinline__ double __shfl_up_double(double & val , unsigned char delta , unsigned char width ){
    return __shfl_up_sync(0xffffffff , val , delta , width);
}
__device__ __forceinline__ double __shfl_down_double(double & val , unsigned char delta , unsigned char width ){
    return __shfl_down_sync(0xffffffff , val , delta , width);
}
__device__ __forceinline__ double __shfl_double(double & val , unsigned char srcLane , unsigned char width){
    return __shfl_sync(0xffffffff , val , srcLane , width);
}

__device__ __forceinline__ double __shfl_xor_double(double & val , unsigned char srcLane , unsigned char width){
    return __shfl_xor_sync(0xffffffff , val , srcLane , width);
}

#else

#define __shfl_up_double(val , delta , witdh) __shfl_up_double_( *( (int2*)(&val) ) , delta , witdh)
__device__ __forceinline__ double __shfl_up_double_(int2 & val , unsigned char delta , unsigned char width ){
    int2 out = *( (int2*)(&val) );
    out.x = __shfl_up(out.x , delta , width);
    out.y = __shfl_up(out.y , delta , width);
    return ( *( (double*)(&out) ) );
}

#define __shfl_down_double(val , delta , witdh) __shfl_down_double_( *( (int2*)(&val) ) , delta , witdh)
__device__ __forceinline__ double __shfl_down_double_(int2 & val , unsigned char delta , unsigned char width ){
    int2 out = *( (int2*)(&val) );
    out.x = __shfl_down(out.x , delta , width);
    out.y = __shfl_down(out.y , delta , width);
    return ( *( (double*)(&out) ) );
}

#define __shfl_double(val , delta , witdh) __shfl_double_( *( (int2*)(&val) ) , delta , witdh)
__device__ __forceinline__ double __shfl_double_(int2 & val , unsigned char srcLane , unsigned char width){
    int2 out = *( (int2*)(&val) );
    out.x = __shfl(out.x , srcLane , width);
    out.y = __shfl(out.y , srcLane , width);
    return ( *( (double*)(&out) ) );
}

#define __shfl_xor_double(val , delta , witdh) __shfl_xor_double_( *( (int2*)(&val) ) , delta , witdh)
__device__ __forceinline__ double __shfl_xor_double_(int2 & val , unsigned char srcLane , unsigned char width){
    int2 out = *( (int2*)(&val) );
    out.x = __shfl_xor(out.x , srcLane , width);
    out.y = __shfl_xor(out.y , srcLane , width);
    return ( *( (double*)(&out) ) );
}

#endif


#ifdef __cplusplus
}
#endif
#endif

#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

uint32_t BlockDimX = 8;
uint32_t BlockDimY = 8;
uint32_t BlockDimZ = 4;

dim3 BlockDim_X = {8, 8, 4};
dim3 BlockDim_Y = {8, 8, 4};
dim3 BlockDim_Z = {8, 8, 4};

dim3 BlockDim = {8, 8, 4};

int MaxThreadsPerBlock;
int MaxBlockDimX;
int MaxBlockDimY;
int MaxBlockDimZ;
int MaxGridDimX;
int MaxGridDimY;
int MaxGridDimZ;


void cuda_commen_init(){
    CUDA_CALL(cudaDeviceGetAttribute(&MaxThreadsPerBlock , cudaDevAttrMaxThreadsPerBlock , 0));
    CUDA_CALL(cudaDeviceGetAttribute(&MaxBlockDimX , cudaDevAttrMaxBlockDimX , 0));
    CUDA_CALL(cudaDeviceGetAttribute(&MaxBlockDimY , cudaDevAttrMaxBlockDimY , 0));
    CUDA_CALL(cudaDeviceGetAttribute(&MaxBlockDimZ , cudaDevAttrMaxBlockDimZ , 0));
    CUDA_CALL(cudaDeviceGetAttribute(&MaxGridDimX , cudaDevAttrMaxGridDimX , 0));
    CUDA_CALL(cudaDeviceGetAttribute(&MaxGridDimY , cudaDevAttrMaxGridDimY , 0));
    CUDA_CALL(cudaDeviceGetAttribute(&MaxGridDimZ , cudaDevAttrMaxGridDimZ , 0));
}

#ifdef __cplusplus
}
#endif

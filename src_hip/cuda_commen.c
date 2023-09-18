#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

uint32_t BlockDimX = 8;
uint32_t BlockDimY = 8;
uint32_t BlockDimZ = 8;

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
    CUDA_CALL(hipDeviceGetAttribute(&MaxThreadsPerBlock , hipDeviceAttributeMaxThreadsPerBlock , 0));
    CUDA_CALL(hipDeviceGetAttribute(&MaxBlockDimX , hipDeviceAttributeMaxBlockDimX , 0));
    CUDA_CALL(hipDeviceGetAttribute(&MaxBlockDimY , hipDeviceAttributeMaxBlockDimY , 0));
    CUDA_CALL(hipDeviceGetAttribute(&MaxBlockDimZ , hipDeviceAttributeMaxBlockDimZ , 0));
    CUDA_CALL(hipDeviceGetAttribute(&MaxGridDimX , hipDeviceAttributeMaxGridDimX , 0));
    CUDA_CALL(hipDeviceGetAttribute(&MaxGridDimY , hipDeviceAttributeMaxGridDimY , 0));
    CUDA_CALL(hipDeviceGetAttribute(&MaxGridDimZ , hipDeviceAttributeMaxGridDimZ , 0));
}

#ifdef __cplusplus
}
#endif

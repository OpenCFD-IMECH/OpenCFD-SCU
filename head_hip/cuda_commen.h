#ifndef _CUDA_HEAD_H
#define _CUDA_HEAD_H

#include "config_parameters.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include "mpi.h"

#ifdef __cplusplus
extern "C"{
#endif


#ifdef DEBUG_MODE
#include "stdio.h"
#define CUDA_CALL(CALL) \
    {\
        hipError_t error;\
        error = CALL;\
        if(error != hipSuccess){\
            printf("In file \"%s\" , line %d ( \"%s\" ) , cuda call failed\n",__FILE__ , __LINE__ , __FUNCTION__);\
            printf("Error code : %s  (%s)\n",hipGetErrorString(error) , hipGetErrorName(error));\
            MPI_Abort(((MPI_Comm)0x44000000) , 1);\
        }\
    }
#else
#define CUDA_CALL(CALL) CALL;
#endif

#ifdef DEBUG_MODE
#include "stdio.h"
#define CUDA_LAUNCH(call)\
    {\
        hipError_t error;\
        call;\
        error = hipGetLastError();\
        if( error != hipSuccess){\
            printf("In file \"%s\" , line %d ( \"%s\" ) , cuda call failed\n",__FILE__ , __LINE__ , __FUNCTION__);\
            printf("Error code : %s  (%s)\n",hipGetErrorString(error) , hipGetErrorName(error));\
            MPI_Abort(((MPI_Comm)0x44000000) , 1);\
        }\
    }
#else
#define CUDA_LAUNCH(call)\
    call;
#endif

typedef struct cudaField_ {
    REAL * ptr;
    unsigned int pitch;
    #ifdef __HIPCC__
    unsigned int dummy0;
    #endif
} cudaField;

typedef struct cudaField_int_ {
    int * ptr;
    unsigned int pitch;
    #ifdef __HIPCC__
    unsigned int dummy0;
    #endif
} cudaField_int;

typedef struct cudaSoA_{
    REAL * ptr;
    unsigned int pitch;
    unsigned int length_Y;
    unsigned int length_Z;
    #ifdef __HIPCC__
    unsigned int dummy0;
    #endif
} cudaSoA;


typedef struct cudaFieldPack_{
    REAL * ptr;
} cudaFieldPack;


extern uint32_t BlockDimX;
extern uint32_t BlockDimY;
extern uint32_t BlockDimZ;

extern dim3 BlockDim_X;
extern dim3 BlockDim_Y;
extern dim3 BlockDim_Z;

extern dim3 BlockDim;

extern int MaxThreadsPerBlock;
extern int MaxBlockDimX;
extern int MaxBlockDimY;
extern int MaxBlockDimZ;
extern int MaxGridDimX;
extern int MaxGridDimY;
extern int MaxGridDimZ;


void cuda_commen_init();


typedef struct cudaJobPackage_
{
    dim3 start;
    dim3 end;
    #ifdef __cplusplus
    __host__ __device__ cudaJobPackage_(){};
    __host__ __device__ cudaJobPackage_(const dim3 & start_in , const dim3 & end_in):start(start_in),end(end_in){};
    __host__ __device__ void setup(const dim3 & start_in , const dim3 & end_in){start = start_in ,end = end_in;};
    #endif
} cudaJobPackage;

inline void jobsize(cudaJobPackage *job , dim3 *size){
    size->x = job->end.x - job->start.x;
    size->y = job->end.y - job->start.y;
    size->z = job->end.z - job->start.z;
}

inline void job2size(cudaJobPackage *job , dim3 *size){
    size->x = job->end.x;
    size->y = job->end.y;
    size->z = job->end.z;
}
inline void job2job(cudaJobPackage * job , cudaJobPackage * out){
    out->start.x = job->start.x;
    out->start.y = job->start.y;
    out->start.z = job->start.z;
    
    out->end.x = job->end.x + job->start.x;
    out->end.y = job->end.y + job->start.y;
    out->end.z = job->end.z + job->start.z;
}

#ifdef __cplusplus
}
#endif
#endif

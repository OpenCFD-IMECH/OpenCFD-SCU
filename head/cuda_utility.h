#ifndef _CUDA_UTILITY_H
#define _CUDA_UTILITY_H

#include "cuda_commen.h"
#include "config_parameters.h"


#ifdef __cplusplus
extern "C"{
#endif

void * malloc_me_d(unsigned int * pitch , unsigned int size_x , unsigned size_y , unsigned size_z);



#define get_Field(Field , x,y,z,...) (*(Field.ptr + (__VA_ARGS__ + x + Field.pitch *(y + (z)*ny_d) ) ) )
#define get_Field_LAP(Field , x,y,z,...) (*(Field.ptr + (__VA_ARGS__ + x + Field.pitch *(y + (z)*ny_2lap_d) ) ) )

void new_cudaField( cudaField ** pField, unsigned int size_x , unsigned int size_y , unsigned int size_z );
void delete_cudaField(cudaField * pField);

void new_cudaField_int( cudaField_int ** pField, unsigned int size_x , unsigned int size_y , unsigned int size_z );
void delete_cudaField_int(cudaField_int * pField);


#define get_SoA(SoA , x,y,z , var,...) (*( SoA.ptr + (__VA_ARGS__ + x + SoA.pitch*(y + ny_d*(z+ (var)*nz_d)))))
#define get_SoA_LAP(SoA , x,y,z , var,...) (*( SoA.ptr + (__VA_ARGS__ + x + SoA.pitch*(y + ny_2lap_d*(z+ (var)*nz_2lap_d)))))
#define access_sp_num_data(ptr, x, y) *(ptr + (x) + (y)*NSPECS) 

void new_cudaSoA( cudaSoA ** pSoA, unsigned int size_x , unsigned int size_y , unsigned int size_z );
void new_cudaSoA_spec( cudaSoA ** pSoA, unsigned int size_x , unsigned int size_y , unsigned int size_z );
void delete_cudaSoA(cudaSoA * pSoA);

void new_cudaFieldPack(cudaFieldPack ** pack , unsigned int size_x , unsigned int size_y , unsigned int size_z);
void delete_cudaFieldPack(cudaFieldPack * pField);


void cal_grid_block_dim(dim3 * pgrid , dim3 * pblock , unsigned int threadx , unsigned int thready , unsigned int threadz ,  unsigned int size_x , unsigned int size_y , unsigned int size_z);



enum {
    H2D = 0,
    D2H = 1
};
void memcpy_All(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z);
void memcpy_spec(REAL *hostPtr, cudaSoA *spec, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z, unsigned int n);
void memcpy_All_int(int *hostPtr, int *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z);
void memcpy_inner(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z);
void memcpy_bound_x(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z);
void memcpy_bound_y(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z);
void memcpy_bound_z(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z);

#ifdef __cplusplus
}
#endif
#endif
#ifndef __UTILITY_H
#define __UTILITY_H
#include <stdlib.h>
#include <stdio.h>
#include "parameters.h"
#include "config_parameters.h"

#define PTR2ARRAY2(ptr,nx,ny) (REAL(*)[ny][nx])(ptr)
#define PTR2ARRAY3(ptr,nx,ny,nz) (REAL(*)[nz][ny][nx])(ptr)
#define MAX(a,b) (a>b? a : b)
#define MIN(a,b) (a<b? a : b)
#define PROCIdx2Num(proc_i , proc_j , proc_k) (proc_i + proc_j*NPX0 + proc_k*NPX0*NPY0)

#ifdef __cplusplus
extern "C"{
#endif

#define malloc_me_Host(p, size) malloc_me_Host_((void**)&p , size ,  __FUNCTION__ , __FILE__ ,__LINE__)
void malloc_me_Host_(void **p, int size , const char * funname , const char * file , int line);

#define malloc_me(size) malloc_me_(size , __FUNCTION__ , __FILE__ ,__LINE__)
void * malloc_me_(int size , const char * funname , const char * file , int line);

#ifdef __cplusplus
}
#endif

#endif
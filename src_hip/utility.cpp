#include "utility.h"
#include "stdio.h"
#include "cuda_commen.h"
#ifdef __cplusplus
extern "C"{
#endif

void malloc_me_Host_(void **p, int size , const char * funname , const char * file , int line){

	hipError_t Status = hipHostMalloc(p, size, hipHostMallocDefault);
    if(Status != hipSuccess){
       printf("Memory allocate error ! Can not allocate enough momory in fun %s ( file %s  , line %d ) , Proc %d\n" , funname ,file,line,my_id);
       MPI_Finalize();
       exit(EXIT_FAILURE);
    }

}

void * malloc_me_(int size , const char * funname , const char * file , int line){
    void * tmp = malloc(size);
    if(tmp == NULL){
       printf("Memory allocate error ! Can not allocate enough momory in fun %s ( file %s  , line %d ) , Proc %d\n" , funname ,file,line,my_id);
       MPI_Finalize();
       exit(EXIT_FAILURE);
    }
    return tmp;
}

#ifdef __cplusplus
}
#endif
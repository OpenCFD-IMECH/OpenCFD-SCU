#include "cuda_runtime.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "parameters.h"
#include "utility.h"


#ifdef __cplusplus
extern "C"{
#endif

#if defined(__HIPCC__)
typedef struct hipExtent Extent_t;
typedef struct hipPos Pos_t;
typedef struct hipPitchedPtr PitchedPtr_t;
typedef struct hipMemcpy3DParms Memcpy3DParms_t;
#define make_Extent(a,b,c) make_hipExtent(a,b,c)
#define make_Pos(a,b,c) make_hipPos(a,b,c)
#define make_PitchedPtr(a,b,c,d) make_hipPitchedPtr(a,b,c,d)
#define Memcpy3D(parm) memcpy3D_me(parm)

#else
typedef struct cudaExtent Extent_t;
typedef struct cudaPos Pos_t;
typedef struct cudaPitchedPtr PitchedPtr_t;
typedef struct cudaMemcpy3DParms Memcpy3DParms_t;
#define make_Extent(a,b,c) make_cudaExtent(a,b,c)
#define make_Pos(a,b,c) make_cudaPos(a,b,c)
#define make_PitchedPtr(a,b,c,d) make_cudaPitchedPtr(a,b,c,d)
#define Memcpy3D(parm) cudaMemcpy3D(parm)
#endif

void *malloc_me_d(unsigned int *pitch, unsigned int size_x, unsigned size_y, unsigned size_z)
{
    PitchedPtr_t ptr;
    Extent_t extent = make_Extent(size_x * sizeof(REAL), size_y, size_z);
    #if defined(__HIPCC__)
    CUDA_CALL( hipMallocPitch(&(ptr.ptr), &(ptr.pitch) , extent.width , extent.height*extent.depth) );
    #else
    CUDA_CALL( cudaMalloc3D(&ptr, extent) );
    #endif

    *pitch = ptr.pitch;
    return ptr.ptr;
}

void new_cudaField(cudaField **pField, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    cudaField tmpField;
    unsigned int pitch;
    void *tmp_ptr;

    tmp_ptr = malloc_me_d(&pitch, size_x , size_y, size_z);

    tmpField.ptr = (REAL *)tmp_ptr;
    pitch /= sizeof(REAL);
    tmpField.pitch = pitch;

    *pField = (cudaField *)malloc(sizeof(cudaField));
    **pField = tmpField;
}

void *malloc_me_int_d(unsigned int *pitch, unsigned int size_x, unsigned size_y, unsigned size_z)
{
    PitchedPtr_t ptr;
    Extent_t extent = make_Extent(size_x * sizeof(int), size_y, size_z);
    #if defined(__HIPCC__)
    CUDA_CALL( hipMallocPitch(&(ptr.ptr), &(ptr.pitch) , extent.width , extent.height*extent.depth) );
    #else
    CUDA_CALL( cudaMalloc3D(&ptr, extent) );
    #endif

    *pitch = ptr.pitch;
    return ptr.ptr;
}

void new_cudaField_int(cudaField_int **pField, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    cudaField_int tmpField;
    unsigned int pitch;
    void *tmp_ptr;

    tmp_ptr = malloc_me_int_d(&pitch, size_x , size_y, size_z);

    tmpField.ptr = (int *)tmp_ptr;
    pitch /= sizeof(int);
    tmpField.pitch = pitch;

    *pField = (cudaField_int *)malloc(sizeof(cudaField_int));
    **pField = tmpField;
}

void delete_cudaField(cudaField *pField)
{
    CUDA_CALL( cudaFree(pField->ptr) );
    free(pField);
    pField = 0;
}

void delete_cudaField_int(cudaField_int *pField)
{
    CUDA_CALL( cudaFree(pField->ptr) );
    free(pField);
    pField = 0;
}

void new_cudaSoA(cudaSoA **pSoA, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    cudaSoA tmpSoA;
    unsigned int pitch;
    void *tmp_ptr;

    tmp_ptr = malloc_me_d(&pitch, size_x, size_y, size_z * NVARS);

    tmpSoA.ptr = (REAL *)tmp_ptr;

    pitch /= sizeof(REAL);
    tmpSoA.pitch = pitch;
    tmpSoA.length_Y = size_y;
    tmpSoA.length_Z = size_z;

    *pSoA = (cudaSoA *)malloc(sizeof(cudaSoA));
    **pSoA = tmpSoA;
}

void new_cudaSoA_spec(cudaSoA **pSoA, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    cudaSoA tmpSoA;
    unsigned int pitch;
    void *tmp_ptr;

    tmp_ptr = malloc_me_d(&pitch, size_x, size_y, size_z * NSPECS);

    tmpSoA.ptr = (REAL *)tmp_ptr;

    pitch /= sizeof(REAL);
    tmpSoA.pitch = pitch;
    tmpSoA.length_Y = size_y;
    tmpSoA.length_Z = size_z;

    *pSoA = (cudaSoA *)malloc(sizeof(cudaSoA));
    **pSoA = tmpSoA;
}

void delete_cudaSoA(cudaSoA *pSoA)
{
    CUDA_CALL( cudaFree(pSoA->ptr) );
    free(pSoA);
    pSoA = 0;
}

void new_cudaFieldPack(cudaFieldPack ** pack , unsigned int size_x , unsigned int size_y , unsigned int size_z){
    int size = size_x*size_y*size_z*sizeof(REAL);
    *pack = (cudaFieldPack*)malloc(sizeof(cudaFieldPack));

    REAL * ptr;
    CUDA_CALL(( cudaMalloc((void**)&ptr , size) ))
    (*pack) -> ptr = ptr;
}

void delete_cudaFieldPack(cudaFieldPack * pack){
    CUDA_CALL(( cudaFree(pack->ptr) ))
    free(pack);
}

/*

{
	struct cudaExtent extent = make_cudaExtent(nx*sizeof(REAL),ny,nz);
    struct cudaPitchedPtr from = make_cudaPitchedPtr(SoA.ptr0 , SoA.pitch , nx , ny);
    struct cudaPitchedPtr to = make_cudaPitchedPtr(buffer , nx*sizeof(REAL) , nx , ny);

    struct cudaMemcpy3DParms parm = {0};
    parm.extent = extent;
    parm.srcPtr = from;
    parm.dstPtr = to;
    parm.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D(&parm);
}

*/

// ================================================================== //
void cal_grid_block_dim(dim3 *pgrid, dim3 *pblock, unsigned int threadx, unsigned int thready, unsigned int threadz, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    pblock->x = threadx;//block的纬度
    pblock->y = thready;
    pblock->z = threadz;

    pgrid->x = (size_x + threadx - 1) / threadx;//三个方向启动block的数目
    pgrid->y = (size_y + thready - 1) / thready;
    pgrid->z = (size_z + threadz - 1) / threadz;
}


#include "assert.h"
#include "mpi.h"
#include "stdio.h"
cudaError_t memcpy3D_me(Memcpy3DParms_t * p){
    assert( p->srcPos.x >=0 );
    assert( p->srcPos.y >=0 );
    assert( p->srcPos.z >=0 );

    assert( p->srcPos.x + p->extent.width  - 1 < p->srcPtr.xsize );
    assert( p->srcPos.y + p->extent.height - 1 < p->srcPtr.ysize );

    assert( p->dstPos.x >=0 );
    assert( p->dstPos.y >=0 );
    assert( p->dstPos.z >=0 );

    assert( p->dstPos.x + p->extent.width  - 1 < p->dstPtr.xsize );
    assert( p->dstPos.y + p->extent.height - 1 < p->dstPtr.ysize );

    int src_size = p->srcPtr.pitch;
    int dst_size = p->dstPtr.pitch;
    char * src = (char*)p->srcPtr.ptr + p->srcPos.x + p->srcPtr.pitch*(p->srcPos.y + p->srcPtr.ysize*p->srcPos.z);
    char * dst = (char*)p->dstPtr.ptr + p->dstPos.x + p->dstPtr.pitch*(p->dstPos.y + p->dstPtr.ysize*p->dstPos.z);

    cudaError_t error;
    int offset_src , offset_dst;
    for(int k=0;k<p->extent.depth;k++){
        offset_src = k*p->srcPtr.ysize;
        offset_dst = k*p->dstPtr.ysize;
        for(int j=0;j<p->extent.height;j++){
            error = cudaMemcpy(dst + p->dstPtr.pitch*(offset_dst+j) , src + p->srcPtr.pitch*(offset_src+j) , p->extent.width , p->kind);
            if(error != cudaSuccess){
                printf("error in memcpy3d_me , ( k = %d , j = %d)\n",k,j);
                printf("In file \"%s\" , line %d ( \"%s\" ) , cuda call failed\n",__FILE__ , __LINE__ , __FUNCTION__);
                printf("Error code : %s  (%s)\n",cudaGetErrorString(error) , cudaGetErrorName(error));
                MPI_Abort(MPI_COMM_WORLD , 1);
            }
        }
    }
    return cudaSuccess;
}

void memcpy_All_int(int *hostPtr, int *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{   
    Extent_t extent = make_Extent(size_x * sizeof(int), size_y, size_z);
    PitchedPtr_t host = make_PitchedPtr(hostPtr, size_x * sizeof(int), size_x*sizeof(int), size_y);
    PitchedPtr_t dev = make_PitchedPtr(devPtr, pitch*sizeof(int), size_x*sizeof(int), size_y);

    Memcpy3DParms_t parm = {0};

    parm.extent = extent;

    if (mode == H2D)
    {
        parm.srcPtr = host;
        parm.dstPtr = dev;
        parm.kind = cudaMemcpyHostToDevice;
    }
    else
    {
        parm.srcPtr = dev;
        parm.dstPtr = host;
        parm.kind = cudaMemcpyDeviceToHost;
    }
    CUDA_CALL( Memcpy3D(&parm) );
}

void memcpy_All(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{   
    Extent_t extent = make_Extent(size_x * sizeof(REAL), size_y, size_z);
    PitchedPtr_t host = make_PitchedPtr(hostPtr, size_x * sizeof(REAL), size_x*sizeof(REAL), size_y);
    PitchedPtr_t dev = make_PitchedPtr(devPtr, pitch*sizeof(REAL), size_x*sizeof(REAL), size_y);

    Memcpy3DParms_t parm = {0};

    parm.extent = extent;

    if (mode == H2D)
    {
        parm.srcPtr = host;
        parm.dstPtr = dev;
        parm.kind = cudaMemcpyHostToDevice;
    }
    else
    {
        parm.srcPtr = dev;
        parm.dstPtr = host;
        parm.kind = cudaMemcpyDeviceToHost;
    }
    CUDA_CALL( Memcpy3D(&parm) );
}

void memcpy_inner(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    // data block is size_x * size_y * size_z

    Extent_t extent = make_Extent( (size_x-2*LAP )* sizeof(REAL), size_y-2*LAP, size_z-2*LAP);
    PitchedPtr_t host = make_PitchedPtr(hostPtr, size_x * sizeof(REAL), size_x*sizeof(REAL), size_y);
    PitchedPtr_t dev = make_PitchedPtr(devPtr, pitch*sizeof(REAL), size_x*sizeof(REAL), size_y);
    Pos_t pos = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
    Memcpy3DParms_t parm = {0};

    parm.extent = extent;

    if (mode == H2D)
    {
        parm.srcPtr = host;
        parm.srcPos = pos;
        parm.dstPtr = dev;
        parm.dstPos = pos;
        parm.kind = cudaMemcpyHostToDevice;
    }
    else
    {
        parm.srcPtr = dev;
        parm.srcPos = pos;
        parm.dstPtr = host;
        parm.dstPos = pos;
        parm.kind = cudaMemcpyDeviceToHost;
    }
    CUDA_CALL( Memcpy3D(&parm) );
}

void memcpy_bound_x(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    Extent_t extent = make_Extent( LAP * sizeof(REAL), size_y-2*LAP, size_z-2*LAP);
    PitchedPtr_t host = make_PitchedPtr(hostPtr, size_x * sizeof(REAL), size_x*sizeof(REAL), size_y);
    PitchedPtr_t dev = make_PitchedPtr(devPtr, pitch*sizeof(REAL), size_x*sizeof(REAL), size_y);

    Memcpy3DParms_t parm_m = {0};
    Memcpy3DParms_t parm_p = {0};

    // 1 - src , 2 - dst
    Pos_t pos_1;
    Pos_t pos_2;

    if (mode == H2D)
    {
        // 1 - src , 2 - dst
        pos_1 = make_Pos(0 , LAP , LAP);
        pos_2 = make_Pos(0 , LAP , LAP);
        parm_m.srcPtr = host;
        parm_m.srcPos = pos_1;
        parm_m.dstPtr = dev;
        parm_m.dstPos = pos_2;
        parm_m.extent = extent;
        parm_m.kind = cudaMemcpyHostToDevice;

        pos_1 = make_Pos(sizeof(REAL)*(size_x-  LAP) , LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*(size_x-  LAP) , LAP , LAP);
        parm_p.srcPtr = host;
        parm_p.srcPos = pos_1;
        parm_p.dstPtr = dev;
        parm_p.dstPos = pos_2;
        parm_p.extent = extent;
        parm_p.kind = cudaMemcpyHostToDevice;
    }
    else
    {
        pos_1 = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
        parm_m.srcPtr = dev;
        parm_m.srcPos = pos_1;
        parm_m.dstPtr = host;
        parm_m.dstPos = pos_2;
        parm_m.extent = extent;
        parm_m.kind = cudaMemcpyDeviceToHost;


        pos_1 = make_Pos(sizeof(REAL)*(size_x-2*LAP) , LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*(size_x-2*LAP) , LAP , LAP);
        parm_p.srcPtr = dev;
        parm_p.srcPos = pos_1;
        parm_p.dstPtr = host;
        parm_p.dstPos = pos_2;
        parm_p.extent = extent;
        parm_p.kind = cudaMemcpyDeviceToHost;
    }
    CUDA_CALL( Memcpy3D(&parm_m) );
    CUDA_CALL( Memcpy3D(&parm_p) );
}

void memcpy_bound_y(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    Extent_t extent = make_Extent(  (size_x - 2*LAP) * sizeof(REAL), LAP, size_z-2*LAP);
    PitchedPtr_t host = make_PitchedPtr(hostPtr, size_x * sizeof(REAL), size_x*sizeof(REAL), size_y);
    PitchedPtr_t dev = make_PitchedPtr(devPtr, pitch*sizeof(REAL), size_x*sizeof(REAL), size_y);

    Memcpy3DParms_t parm_m = {0};
    Memcpy3DParms_t parm_p = {0};

    Pos_t pos_1;
    Pos_t pos_2;
    if (mode == H2D)
    {
        pos_1 = make_Pos(sizeof(REAL)*LAP , 0 , LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , 0 , LAP);
        parm_m.srcPtr = host;
        parm_m.srcPos = pos_1;
        parm_m.dstPtr = dev;
        parm_m.dstPos = pos_2;
        parm_m.extent = extent;
        parm_m.kind = cudaMemcpyHostToDevice;

        pos_1 = make_Pos(sizeof(REAL)*LAP , size_y-LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , size_y-LAP , LAP);
        parm_p.srcPtr = host;
        parm_p.srcPos = pos_1;
        parm_p.dstPtr = dev;
        parm_p.dstPos = pos_2;
        parm_p.extent = extent;
        parm_p.kind = cudaMemcpyHostToDevice;
    }
    else
    {
        pos_1 = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
        parm_m.srcPtr = dev;
        parm_m.srcPos = pos_1;
        parm_m.dstPtr = host;
        parm_m.dstPos = pos_2;
        parm_m.extent = extent;
        parm_m.kind = cudaMemcpyDeviceToHost;

        pos_1 = make_Pos(sizeof(REAL)*LAP , size_y-2*LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , size_y-2*LAP , LAP);
        parm_p.srcPtr = dev;
        parm_p.srcPos = pos_1;
        parm_p.dstPtr = host;
        parm_p.dstPos = pos_2;
        parm_p.extent = extent;
        parm_p.kind = cudaMemcpyDeviceToHost;
    }
    CUDA_CALL( Memcpy3D(&parm_m) );
    CUDA_CALL( Memcpy3D(&parm_p) );
}

void memcpy_bound_z(REAL *hostPtr, REAL *devPtr, unsigned int pitch, int mode, unsigned int size_x, unsigned int size_y, unsigned int size_z)
{
    Extent_t extent = make_Extent(  (size_x - 2*LAP) * sizeof(REAL), size_y - 2*LAP, LAP);
    PitchedPtr_t host = make_PitchedPtr(hostPtr, size_x * sizeof(REAL), size_x*sizeof(REAL), size_y);
    PitchedPtr_t dev = make_PitchedPtr(devPtr, pitch*sizeof(REAL), size_x*sizeof(REAL), size_y);

    Memcpy3DParms_t parm_m = {0};
    Memcpy3DParms_t parm_p = {0};

    Pos_t pos_1;
    Pos_t pos_2;
    if (mode == H2D)
    {
        pos_1 = make_Pos(sizeof(REAL)*LAP , LAP , 0);
        pos_2 = make_Pos(sizeof(REAL)*LAP , LAP , 0);
        parm_m.srcPtr = host;
        parm_m.srcPos = pos_1;
        parm_m.dstPtr = dev;
        parm_m.dstPos = pos_2;
        parm_m.extent = extent;
        parm_m.kind = cudaMemcpyHostToDevice;

        pos_1 = make_Pos(sizeof(REAL)*LAP , LAP , size_z-LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , LAP , size_z-LAP);
        parm_p.srcPtr = host;
        parm_p.srcPos = pos_1;
        parm_p.dstPtr = dev;
        parm_p.dstPos = pos_2;
        parm_p.extent = extent;
        parm_p.kind = cudaMemcpyHostToDevice;
    }
    else
    {
        pos_1 = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , LAP , LAP);
        parm_m.srcPtr = dev;
        parm_m.srcPos = pos_1;
        parm_m.dstPtr = host;
        parm_m.dstPos = pos_2;
        parm_m.extent = extent;
        parm_m.kind = cudaMemcpyDeviceToHost;

        pos_1 = make_Pos(sizeof(REAL)*LAP , LAP , size_z-2*LAP);
        pos_2 = make_Pos(sizeof(REAL)*LAP , LAP , size_z-2*LAP);
        parm_p.srcPtr = dev;
        parm_p.srcPos = pos_1;
        parm_p.dstPtr = host;
        parm_p.dstPos = pos_2;
        parm_p.extent = extent;
        parm_p.kind = cudaMemcpyDeviceToHost;
    }
    CUDA_CALL( Memcpy3D(&parm_m) );
    CUDA_CALL( Memcpy3D(&parm_p) );
}

#ifdef __cplusplus
}
#endif

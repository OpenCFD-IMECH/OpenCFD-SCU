/*OpenCFD ver 1.4, CopyRight by Li Xinliang, LNM, Institute of Mechanics, CAS, Beijing, Email: lixl@lnm.imech.ac.cn
MPI Subroutines, such as computational domain partation, MPI message send and recv   
只支持N_MSG_SIZE=0, -2  两种通信方式 
*/

#include "mpi.h"
#include "OCFD_mpi.h"
#include "parameters.h"
#include "parameters_d.h"

#include "cuda_commen.h"
#include "cuda_utility.h"
#include "OCFD_mpi_dev.h"


#ifdef __cplusplus
extern "C"{
#endif

// -------------------------------------------------------------------------------------
// Message send and recv at inner boundary (or 'MPI boundary')
void exchange_boundary_xyz_dev(REAL *hostptr , cudaField * devptr)
{
	exchange_boundary_x_dev(hostptr , devptr, Iperiodic[0]);
	exchange_boundary_y_dev(hostptr , devptr, Iperiodic[1]);
	exchange_boundary_z_dev(hostptr , devptr, Iperiodic[2]);
}
// ----------------------------------------------------------------------------------------
void exchange_boundary_x_dev(REAL *hostptr , cudaField * devptr , int Iperiodic1)
{
	exchange_boundary_x_standard_dev(hostptr , devptr, Iperiodic1);
}
// -----------------------------------------------------------------------------------------------
void exchange_boundary_y_dev(REAL *hostptr, cudaField * devptr , int Iperiodic1)
{
	exchange_boundary_y_standard_dev(hostptr , devptr, Iperiodic1);
}
// -----------------------------------------------------------------------------------------------
void exchange_boundary_z_dev(REAL *hostptr , cudaField * devptr , int Iperiodic1)
{
	exchange_boundary_z_standard_dev(hostptr , devptr, Iperiodic1);
}



void exchange_boundary_x_standard_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1)
{
	memcpy_bound_x(hostptr , devptr->ptr , devptr->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
	exchange_boundary_x_standard(hostptr , Iperiodic1);
	memcpy_bound_x(hostptr , devptr->ptr , devptr->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
}
// ------------------------------------------------------
void exchange_boundary_y_standard_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1)
{
	memcpy_bound_y(hostptr , devptr->ptr , devptr->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
	exchange_boundary_y_standard(hostptr , Iperiodic1);
	memcpy_bound_y(hostptr , devptr->ptr , devptr->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
}
// ------------------------------------------------------------
void exchange_boundary_z_standard_dev(REAL *hostptr, cudaField * devptr , int Iperiodic1)
{
	memcpy_bound_z(hostptr , devptr->ptr , devptr->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
	exchange_boundary_z_standard(hostptr , Iperiodic1);
	memcpy_bound_z(hostptr , devptr->ptr , devptr->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
}


/* ===================================================================================================== */


static char mpi_dev_mem_initialized = 0;
static cudaFieldPack *b_xm , *b_xp;
static cudaFieldPack *b_ym , *b_yp;
static cudaFieldPack *b_zm , *b_zp;

void mpi_dev_buffer_attach(){
    // N in bytes
    new_cudaFieldPack(&b_xm , LAP , ny , nz);
    new_cudaFieldPack(&b_xp , LAP , ny , nz);
    new_cudaFieldPack(&b_ym , nx , LAP , nz);
    new_cudaFieldPack(&b_yp , nx , LAP , nz);
    new_cudaFieldPack(&b_zm , nx , ny , LAP);
    new_cudaFieldPack(&b_zp , nx , ny , LAP);
}
void mpi_dev_buffer_detach(){
    delete_cudaFieldPack(b_xm);
    delete_cudaFieldPack(b_xp);
    delete_cudaFieldPack(b_ym);
    delete_cudaFieldPack(b_yp);
    delete_cudaFieldPack(b_zm);
    delete_cudaFieldPack(b_zp);
}
// unsigned int buffer_align_length = 32; // bytes
// void dev_buffer_malloc(unsigned int N){
//     // N in bytes
//     int n = N/buffer_align_length + 1;
//     cudaMalloc(n*buffer_align_length);
// }
// void dev_buffer_free(){

// }
// void new_cudaField_buffer(){
    
// }
// void delete_cudaField_buffer(){

// }

void opencfd_mem_init_mpi_dev(){
    if(mpi_dev_mem_initialized == 0){
        mpi_dev_mem_initialized = 1;
        mpi_dev_buffer_attach();
    }
}
void opencfd_mem_finalize_mpi_dev(){
    if(mpi_dev_mem_initialized == 1){
        mpi_dev_mem_initialized = 0;
        mpi_dev_buffer_detach();
    }
}

void exchange_boundary_xyz_packed_dev(REAL *hostptr , cudaField * devptr)
{
	exchange_boundary_x_packed_dev(hostptr , devptr, Iperiodic[0]);
	exchange_boundary_y_packed_dev(hostptr , devptr, Iperiodic[1]);
	exchange_boundary_z_packed_dev(hostptr , devptr, Iperiodic[2]);
}

void exchange_boundary_xyz_Async_packed_dev(REAL *hostptr , cudaField * devptr , cudaStream_t *stream)
{
	exchange_boundary_x_Async_packed_dev(hostptr , devptr, Iperiodic[0], stream);
	exchange_boundary_y_Async_packed_dev(hostptr , devptr, Iperiodic[1], stream);
	exchange_boundary_z_Async_packed_dev(hostptr , devptr, Iperiodic[2], stream);
}


__global__ void cudaFieldBoundaryPack_kernel(cudaField data , cudaFieldPack pack , cudaJobPackage job){
    // eyes on cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	if(x < job.end.x && y < job.end.y && z < job.end.z){

        unsigned int pos = x + job.end.x*(y + job.end.y*z);
        x += job.start.x;
        y += job.start.y;
        z += job.start.z;
        *(pack.ptr + pos) = get_Field_LAP(data , x,y,z);
        
    }
}
__global__ void cudaFieldBoundaryUnpack_kernel(cudaField data , cudaFieldPack pack , cudaJobPackage job){
    // eyes on cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	if(x < job.end.x && y < job.end.y && z < job.end.z){

        unsigned int pos = x + job.end.x*(y + job.end.y*z);
        x += job.start.x;
        y += job.start.y;
        z += job.start.z;
        get_Field_LAP(data , x,y,z) = *(pack.ptr + pos);
        
    }
}


void cudaFieldBoundaryPack(cudaField * data , cudaFieldPack * pack, cudaJobPackage job_in)
{
    // job_in , to packed data , with LAP
    // job_in.start , job_in.size
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , job_in.end.x , job_in.end.y , job_in.end.z);
    CUDA_LAUNCH(( cudaFieldBoundaryPack_kernel<<<griddim,blockdim>>>(*data , *pack , job_in) ))
}

void cudaFieldBoundaryUnpack(cudaField * data , cudaFieldPack * pack , cudaJobPackage job_in){
    // job_in , to packed data , with LAP
    // job_in.start , job_in.size
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , job_in.end.x , job_in.end.y , job_in.end.z);
    CUDA_LAUNCH(( cudaFieldBoundaryUnpack_kernel<<<griddim,blockdim>>>(*data , *pack , job_in) ))
}

void exchange_boundary_x_packed_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1)
{   
    cudaFieldPack * pack;
    MPI_Status status;
    int size = LAP*ny*nz;
    pack = b_xm;
    cudaJobPackage job(dim3(LAP,LAP,LAP),dim3(LAP,ny,nz));

    if(npx != 0 || Iperiodic1 == 1){
        cudaFieldBoundaryPack(devptr , pack ,job);
        CUDA_CALL(( cudaMemcpy(pack_send_x , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost) ))
    }
    MPI_Sendrecv(pack_send_x , size , OCFD_DATA_TYPE , ID_XM1 , 1 , pack_recv_x , size , OCFD_DATA_TYPE , ID_XP1 , 1 , MPI_COMM_WORLD , &status);
	if (npx != NPX0 - 1 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpy(pack->ptr , pack_recv_x , size*sizeof(REAL) , cudaMemcpyHostToDevice) ))
        job.start.x = nx_lap;
        cudaFieldBoundaryUnpack(devptr, pack ,job);
    }
    


    if(npx != NPX0 - 1 || Iperiodic1 == 1){
        job.start.x = nx;
        cudaFieldBoundaryPack(devptr , pack ,job);
        CUDA_CALL(( cudaMemcpy(pack_send_x , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost) ))
    }
    MPI_Sendrecv(pack_send_x , size , OCFD_DATA_TYPE , ID_XP1 , 1 , pack_recv_x , size , OCFD_DATA_TYPE , ID_XM1 , 1 , MPI_COMM_WORLD , &status);
	if (npx != 0 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpy(pack->ptr , pack_recv_x , size*sizeof(REAL) , cudaMemcpyHostToDevice) ))
        job.start.x = 0;
        cudaFieldBoundaryUnpack(devptr, pack ,job);
    }
    
}


void exchange_boundary_y_packed_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1)
{   
    cudaFieldPack * pack;
    MPI_Status status;
    int size = LAP*nx*nz;
    pack = b_ym;
    cudaJobPackage job(dim3(LAP,LAP,LAP),dim3(nx , LAP ,nz));

    if(npy != 0 || Iperiodic1 == 1){
        cudaFieldBoundaryPack(devptr , pack ,job);
        CUDA_CALL(( cudaMemcpy(pack_send_y , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost) ))
    }
    MPI_Sendrecv(pack_send_y , size , OCFD_DATA_TYPE , ID_YM1 , 1 , pack_recv_y , size , OCFD_DATA_TYPE , ID_YP1 , 1 , MPI_COMM_WORLD , &status);
	if (npy != NPY0 - 1 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpy(pack->ptr , pack_recv_y , size*sizeof(REAL) , cudaMemcpyHostToDevice) ))
        job.start.y = ny_lap;
        cudaFieldBoundaryUnpack(devptr, pack ,job);
    }
    


    if(npy != NPY0 - 1 || Iperiodic1 == 1){
        job.start.y = ny;
        cudaFieldBoundaryPack(devptr , pack ,job);
        CUDA_CALL(( cudaMemcpy(pack_send_y , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost) ))
    }
    MPI_Sendrecv(pack_send_y , size , OCFD_DATA_TYPE , ID_YP1 , 1 , pack_recv_y , size , OCFD_DATA_TYPE , ID_YM1 , 1 , MPI_COMM_WORLD , &status);
	if (npy != 0 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpy(pack->ptr , pack_recv_y , size*sizeof(REAL) , cudaMemcpyHostToDevice) ))
        job.start.y = 0;
        cudaFieldBoundaryUnpack(devptr, pack ,job);
    }

}


void exchange_boundary_z_packed_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1)
{   
    cudaFieldPack * pack;
    MPI_Status status;
    int size = LAP*nx*ny;
    pack = b_zm;
    cudaJobPackage job(dim3(LAP,LAP,LAP),dim3(nx,ny,LAP));

    if(npz != 0 || Iperiodic1 == 1){
        cudaFieldBoundaryPack(devptr , pack ,job);
        CUDA_CALL(( cudaMemcpy(pack_send_z , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost) ))
    }
    MPI_Sendrecv(pack_send_z , size , OCFD_DATA_TYPE , ID_ZM1 , 1 , pack_recv_z , size , OCFD_DATA_TYPE , ID_ZP1 , 1 , MPI_COMM_WORLD , &status);
	if (npz != NPZ0 - 1 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpy(pack->ptr , pack_recv_z , size*sizeof(REAL) , cudaMemcpyHostToDevice) ))
        job.start.z = nz_lap;
        cudaFieldBoundaryUnpack(devptr, pack ,job);
    }
    

    if(npz != NPZ0 - 1 || Iperiodic1 == 1){
        job.start.z = nz;
        cudaFieldBoundaryPack(devptr , pack ,job);
        CUDA_CALL(( cudaMemcpy(pack_send_z , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost) ))
    }
    MPI_Sendrecv(pack_send_z , size , OCFD_DATA_TYPE , ID_ZP1 , 1 , pack_recv_z , size , OCFD_DATA_TYPE , ID_ZM1 , 1 , MPI_COMM_WORLD , &status);
	if (npz != 0 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpy(pack->ptr , pack_recv_z , size*sizeof(REAL) , cudaMemcpyHostToDevice) ))
        job.start.z = 0;
        cudaFieldBoundaryUnpack(devptr, pack ,job);
    }

}

void cudaFieldBoundaryPack_Async(cudaField * data , cudaFieldPack * pack, cudaJobPackage job_in, cudaStream_t *stream)
{
    // job_in , to packed data , with LAP
    // job_in.start , job_in.size
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , job_in.end.x , job_in.end.y , job_in.end.z);
    CUDA_LAUNCH(( cudaFieldBoundaryPack_kernel<<<griddim,blockdim,0,*stream>>>(*data , *pack , job_in) ))
}

void cudaFieldBoundaryUnpack_Async(cudaField * data , cudaFieldPack * pack , cudaJobPackage job_in, cudaStream_t *stream){
    // job_in , to packed data , with LAP
    // job_in.start , job_in.size
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , job_in.end.x , job_in.end.y , job_in.end.z);
    CUDA_LAUNCH(( cudaFieldBoundaryUnpack_kernel<<<griddim,blockdim,0,*stream>>>(*data , *pack , job_in) ))
}

// 假设 ， 仅仅交换边界
void exchange_boundary_x_Async_packed_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1 , cudaStream_t *stream)
{   
    cudaFieldPack * pack;
    MPI_Status status;
    int size = LAP*ny*nz;
    pack = b_xm;
    cudaJobPackage job(dim3(LAP,LAP,LAP),dim3(LAP,ny,nz));

    if(npx != 0 || Iperiodic1 == 1){
        cudaFieldBoundaryPack_Async(devptr , pack ,job, stream);
        CUDA_CALL(( cudaMemcpyAsync(pack_send_x , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost, *stream) ))
    }
    cudaStreamSynchronize(*stream);
    MPI_Sendrecv(pack_send_x , size , OCFD_DATA_TYPE , ID_XM1 , 1 , pack_recv_x , size , OCFD_DATA_TYPE , ID_XP1 , 1 , MPI_COMM_WORLD , &status);
	if (npx != NPX0 - 1 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpyAsync(pack->ptr , pack_recv_x , size*sizeof(REAL) , cudaMemcpyHostToDevice, *stream) ))
        job.start.x = nx_lap;
        cudaFieldBoundaryUnpack_Async(devptr, pack ,job, stream);
    }
    


    if(npx != NPX0 - 1 || Iperiodic1 == 1){
        job.start.x = nx;
        cudaFieldBoundaryPack_Async(devptr , pack ,job, stream);
        CUDA_CALL(( cudaMemcpyAsync(pack_send_x , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost, *stream) ))
    }
    cudaStreamSynchronize(*stream);
    MPI_Sendrecv(pack_send_x , size , OCFD_DATA_TYPE , ID_XP1 , 1 , pack_recv_x , size , OCFD_DATA_TYPE , ID_XM1 , 1 , MPI_COMM_WORLD , &status);
	if (npx != 0 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpyAsync(pack->ptr , pack_recv_x , size*sizeof(REAL) , cudaMemcpyHostToDevice, *stream) ))
        job.start.x = 0;
        cudaFieldBoundaryUnpack_Async(devptr, pack ,job, stream);
    }
    
}

void exchange_boundary_y_Async_packed_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1 , cudaStream_t *stream)
{   
    cudaFieldPack * pack;
    MPI_Status status;
    int size = LAP*nx*nz;
    pack = b_ym;
    cudaJobPackage job(dim3(LAP,LAP,LAP),dim3(nx , LAP ,nz));

    if(npy != 0 || Iperiodic1 == 1){
        cudaFieldBoundaryPack_Async(devptr , pack ,job, stream);
        CUDA_CALL(( cudaMemcpyAsync(pack_send_y , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost, *stream) ))
    }
    cudaStreamSynchronize(*stream);
    MPI_Sendrecv(pack_send_y , size , OCFD_DATA_TYPE , ID_YM1 , 1 , pack_recv_y , size , OCFD_DATA_TYPE , ID_YP1 , 1 , MPI_COMM_WORLD , &status);
	if (npy != NPY0 - 1 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpyAsync(pack->ptr , pack_recv_y , size*sizeof(REAL) , cudaMemcpyHostToDevice, *stream) ))
        job.start.y = ny_lap;
        cudaFieldBoundaryUnpack_Async(devptr, pack ,job, stream);
    }
    


    if(npy != NPY0 - 1 || Iperiodic1 == 1){
        job.start.y = ny;
        cudaFieldBoundaryPack_Async(devptr , pack ,job, stream);
        CUDA_CALL(( cudaMemcpyAsync(pack_send_y , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost, *stream) ))
    }
    cudaStreamSynchronize(*stream);
    MPI_Sendrecv(pack_send_y , size , OCFD_DATA_TYPE , ID_YP1 , 1 , pack_recv_y , size , OCFD_DATA_TYPE , ID_YM1 , 1 , MPI_COMM_WORLD , &status);
	if (npy != 0 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpyAsync(pack->ptr , pack_recv_y , size*sizeof(REAL) , cudaMemcpyHostToDevice, *stream) ))
        job.start.y = 0;
        cudaFieldBoundaryUnpack_Async(devptr, pack ,job, stream);
    }

}

void exchange_boundary_z_Async_packed_dev(REAL *hostptr , cudaField * devptr, int Iperiodic1 , cudaStream_t *stream)
{   
    cudaFieldPack * pack;
    MPI_Status status;
    int size = LAP*nx*ny;
    pack = b_zm;
    cudaJobPackage job(dim3(LAP,LAP,LAP),dim3(nx,ny,LAP));

    if(npz != 0 || Iperiodic1 == 1){
        cudaFieldBoundaryPack_Async(devptr , pack ,job, stream);
        CUDA_CALL(( cudaMemcpyAsync(pack_send_z , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost, *stream) ))
    }
    cudaStreamSynchronize(*stream);
    MPI_Sendrecv(pack_send_z , size , OCFD_DATA_TYPE , ID_ZM1 , 1 , pack_recv_z , size , OCFD_DATA_TYPE , ID_ZP1 , 1 , MPI_COMM_WORLD , &status);
	if (npz != NPZ0 - 1 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpyAsync(pack->ptr , pack_recv_z , size*sizeof(REAL) , cudaMemcpyHostToDevice, *stream) ))
        job.start.z = nz_lap;
        cudaFieldBoundaryUnpack_Async(devptr, pack ,job, stream);
    }
    


    if(npz != NPZ0 - 1 || Iperiodic1 == 1){
        job.start.z = nz;
        cudaFieldBoundaryPack_Async(devptr , pack ,job, stream);
        CUDA_CALL(( cudaMemcpyAsync(pack_send_z , pack->ptr , size*sizeof(REAL) , cudaMemcpyDeviceToHost, *stream) ))
    }
    cudaStreamSynchronize(*stream);
    MPI_Sendrecv(pack_send_z , size , OCFD_DATA_TYPE , ID_ZP1 , 1 , pack_recv_z , size , OCFD_DATA_TYPE , ID_ZM1 , 1 , MPI_COMM_WORLD , &status);
	if (npz != 0 || Iperiodic1 == 1){
        CUDA_CALL(( cudaMemcpyAsync(pack->ptr , pack_recv_z , size*sizeof(REAL) , cudaMemcpyHostToDevice, *stream) ))
        job.start.z = 0;
        cudaFieldBoundaryUnpack_Async(devptr, pack ,job, stream);
    }

}


/* 

    switch(dir){
        case MPI_X_DIR : {
            MPI_Sendrecv(pack_send , size , OCFD_DATA_TYPE , ID_XM1 , 1 , pack_recv , size , OCFD_DATA_TYPE , ID_XP1 , 1 , MPI_COMM_WORKD , &status);
            break;
        }
        case MPI_Y_DIR : {
            MPI_Sendrecv(pack_send , size , OCFD_DATA_TYPE , ID_XM1 , 1 , pack_recv , size , OCFD_DATA_TYPE , ID_XP1 , 1 , MPI_COMM_WORKD , &status);
            break;
        }
        case MPI_Z_DIR : {
            MPI_Sendrecv(pack_send , size , OCFD_DATA_TYPE , ID_XM1 , 1 , pack_recv , size , OCFD_DATA_TYPE , ID_XP1 , 1 , MPI_COMM_WORKD , &status);
            break;
        }
    }

*/



#ifdef __cplusplus
}
#endif

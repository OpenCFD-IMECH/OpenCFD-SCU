#ifndef __OCFD_MPI_DEV_H
#define __OCFD_MPI_DEV_H
#include "cuda_commen.h"


#ifdef __cplusplus
extern "C" {
#endif

void exchange_boundary_xyz_dev(REAL *hostptr, cudaField * devptr);
void exchange_boundary_x_dev(REAL *hostptr, cudaField * devptr, int Iperiodic1);
void exchange_boundary_y_dev(REAL *hostptr, cudaField * devptr, int Iperiodic1);
void exchange_boundary_z_dev(REAL *hostptr, cudaField * devptr, int Iperiodic1);


void exchange_boundary_x_standard_dev(REAL *hostptr, cudaField * devptr, int Iperiodic1);
void exchange_boundary_y_standard_dev(REAL *hostptr, cudaField * devptr, int Iperiodic1);
void exchange_boundary_z_standard_dev(REAL *hostptr, cudaField * devptr, int Iperiodic1);

void opencfd_mem_init_mpi_dev();
void opencfd_mem_finalize_mpi_dev();

void exchange_boundary_xyz_Async_packed_dev(cudaField * devptr , cudaStream_t *stream);
void exchange_boundary_x_Async_packed_dev(cudaField * devptr, int Iperiodic1 , cudaStream_t *stream);
void exchange_boundary_y_Async_packed_dev(cudaField * devptr, int Iperiodic1 , cudaStream_t *stream);
void exchange_boundary_z_Async_packed_dev(cudaField * devptr, int Iperiodic1 , cudaStream_t *stream);

void exchange_boundary_xyz_packed_dev(cudaField * devptr);
void exchange_boundary_x_packed_dev(cudaField * devptr, int Iperiodic1);
void exchange_boundary_y_packed_dev(cudaField * devptr, int Iperiodic1);
void exchange_boundary_z_packed_dev(cudaField * devptr, int Iperiodic1);

void exchange_spec_boundary_xyz_packed_dev(cudaSoA * devptr);
void exchange_spec_boundary_x_packed_dev(cudaSoA * devptr, int Iperiodic1);
void exchange_spec_boundary_y_packed_dev(cudaSoA * devptr, int Iperiodic1);
void exchange_spec_boundary_z_packed_dev(cudaSoA * devptr, int Iperiodic1);

#ifdef __cplusplus
}
#endif
#endif
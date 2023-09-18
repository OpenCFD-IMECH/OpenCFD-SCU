#ifndef __OCFD_MPI_H
#define __OCFD_MPI_H
#include "parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void opencfd_mem_init_mpi();
void opencfd_mem_finalize_mpi();
void mpi_init(int * , char ***);
void mpi_finalize();
void part();
static inline int idx2int(int i,int j,int k){
    return(i + j*(nx+2*LAP) + k*(nx+2*LAP)*(ny+2*LAP));
}
int my_mod1(int i,int n);
void New_MPI_datatype();
void get_i_node(int i_global,int * node_i, int * i_local);
void get_j_node(int j_global,int * node_j, int * j_local);
void get_k_node(int k_global,int * node_k, int * k_local);
int get_id(int npx1,int npy1,int npz1);


void exchange_boundary_xyz(REAL *pf);
void exchange_boundary_x(REAL *pf,int Iperiodic1);
void exchange_boundary_y(REAL *pf,int Iperiodic1);
void exchange_boundary_z(REAL *pf,int Iperiodic1);


void exchange_boundary_x_standard(REAL *pf, int Iperiodic1);
void exchange_boundary_y_standard(REAL *pf, int Iperiodic1);
void exchange_boundary_z_standard(REAL *pf, int Iperiodic1);


void exchange_boundary_x_deftype(REAL * pf);
void exchange_boundary_y_deftype(REAL * pf);
void exchange_boundary_z_deftype(REAL * pf);

#ifdef __cplusplus
}
#endif 
#endif
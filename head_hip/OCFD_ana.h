#ifndef __OCFD_ANA_H
#define __OCFD_ANA_H
#include "cuda_commen.h"
#include "parameters.h"
#include "parameters_d.h"
#include "cuda_utility.h"
#include "math.h"
#include "OCFD_Schemes_hybrid_auto.h"
#include "OCFD_Schemes_Choose.h"
#include "OCFD_IO_mpi.h"
#include "utility.h"
#include "OCFD_IO.h"

#ifdef __cplusplus
extern "C"{
#endif

void ana_residual(cudaField PE_d, REAL *E0);
void ana_Jac();
void OCFD_ana(int style, int ID);
void ana_NAN_and_NT();
void init_time_average();

void get_inner(cudaField x1, cudaField x2);

#ifdef __cplusplus
}
#endif
#endif
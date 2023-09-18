#ifndef __OCFD_FILTERINT_H
#define __OCFD_FILTERINT_H
#include "parameters.h"
#include "cuda_commen.h"


#ifdef __cplusplus
extern "C"{
#endif
void filtering(REAL *pf,REAL *pf0,REAL *pp);

void filter_x3d(REAL *pf, REAL *pf0, REAL s0, int ib, int ie, int jb, int je, int kb, int ke);
void filter_y3d(REAL *pf, REAL *pf0, REAL s0, int ib, int ie, int jb, int je, int kb, int ke);
void filter_z3d(REAL *pf, REAL *pf0, REAL s0, int ib, int ie, int jb, int je, int kb, int ke);

void filter_x3d_shock(cudaSoA *pf, cudaSoA *pf0, cudaField *pp, REAL s0, REAL rth, int ib, int ie, int jb, int je, int kb, int ke, int IF_Filter);
void filter_y3d_shock(cudaSoA *pf, cudaSoA *pf0, cudaField *pp, REAL s0, REAL rth, int ib, int ie, int jb, int je, int kb, int ke, int IF_Filter);
void filter_z3d_shock(cudaSoA *pf, cudaSoA *pf0, cudaField *pp, REAL s0, REAL rth, int ib, int ie, int jb, int je, int kb, int ke, int IF_Filter);

void set_para_filtering();
#ifdef __cplusplus
}
#endif
#endif
#ifndef __OCFD_BOUNDARY_INIT_H
#define __OCFD_BOUNDARY_INIT_H
#include "parameters.h"
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

// used in boudary_liftbody*********************************************************************
extern REAL *pu2d_inlet; //[5][nz][ny]
extern REAL *pu2d_upper; //[5][ny][nx]
        
extern REAL * pv_dist_wall; // [ny][nx]
extern REAL *pv_dist_coeff; // [3][ny][nx]
extern REAL * pu_dist_upper; // [ny][nx]

extern cudaField *pu2d_inlet_d; //[5][nz][ny]
extern cudaField *pu2d_upper_d; //[5][ny][nx]
//extern cudaField *pv_dist_wall_d; // [ny][nx]
extern cudaField *pv_dist_coeff_d; // [3][ny][nx]
extern cudaField *pu_dist_upper_d; // [ny][nx]

void bc_parameter();

void bc_user_Liftbody3d_init();
//**********************************************************************************************

// used in boundary_compressible_conner*********************************************************
extern REAL *pub1; // [ny][4]
extern REAL *pfx; // [nx]
extern REAL *pgz; // [nz]
extern REAL *TM; // [MTMAX]
extern REAL *fait; // [MTMAX]
extern REAL SLZ;

extern cudaField *pub1_d; // [ny][4]
extern cudaField *pfx_d; // [nx]
extern cudaField *pgz_d; // [nz]

void bc_user_Compression_conner_init();
//**********************************************************************************************


void get_fait_multifrequancy(int MT_MAX);

void get_xy_blow_suction_multiwave(int NX, int MZ_MAX, REAL *xx,
REAL *fx, REAL *gz, REAL DIST_BEGIN, REAL DIST_END);

void get_xs_blow_suction_multiwave(int NX, int NZ, int MZ_MAX, REAL *xx,
REAL *zz, REAL SL, REAL *fx, REAL *gz, REAL DIST_BEGIN, REAL DIST_END);


#ifdef __cplusplus
}
#endif
#endif

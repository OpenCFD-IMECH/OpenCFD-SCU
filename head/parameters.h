#ifndef __PARAMETER_H
#define __PARAMETER_H
#include "mpi.h"
#include "pthread.h"
#include "config_parameters.h"
#include "OCFD_Schemes_hybrid_auto.h"
#include "stdarg.h"

#ifdef __cplusplus
extern "C"{
#endif

// ------For Doubleprecision  (real*8)------------------------------------------------------------------
extern int OCFD_REAL_KIND;
extern MPI_Datatype OCFD_DATA_TYPE;   //double precison computing
//  ------For Single precision (real*4)-----------------------
// typedef float REAL;
// int OCFD_REAL_KIND=4
// MPI_Datatype OCFD_DATA_TYPE;             //  single precision computing
// ===========  Parameters for MPI ==========================================================----------------
 


// -----constant-----
extern REAL Re , Pr , Ama , Gamma , Ref_T , epsl_SW , PI;
extern REAL Cv , Cp , Tsb , amu_C0;
extern REAL split_C1 , split_C3, tmp0;
extern REAL vis_flux_init_c;

// --------MPI-------------------
extern int my_id,npx,npy,npz;  //全局即方向id
extern int NPX0 , NPY0 , NPZ0; // proc number on each direction
extern int ID_XP1,ID_XM1,ID_YP1,ID_YM1,ID_ZP1,ID_ZM1; //邻居全局id

extern MPI_Status status;
extern MPI_Comm MPI_COMM_X,MPI_COMM_Y,MPI_COMM_Z,MPI_COMM_XY,MPI_COMM_XZ,MPI_COMM_YZ;
extern MPI_Datatype TYPE_LAPX1,TYPE_LAPY1,TYPE_LAPZ1,TYPE_LAPX2,TYPE_LAPY2,TYPE_LAPZ2;

extern int *i_offset,*j_offset,*k_offset,*i_nn,*j_nn,*k_nn; //某个方向的分块信息
extern int MSG_BLOCK_SIZE;

extern unsigned int nx,ny,nz; // 某方向所处理的个数
extern unsigned int NX_GLOBAL,NY_GLOBAL,NZ_GLOBAL;
extern unsigned int nx_lap,ny_lap,nz_lap;
extern unsigned int nx_2lap,ny_2lap,nz_2lap;

extern int Stream_MODE; //Stream 模式
extern int TEST;
extern pthread_t* thread_handles;
// --------------------------------------------------------------------------------------------------------
extern REAL dt,end_time,tt;
extern REAL cpu_time;
extern int Istep , end_step;

// -----------Analysis and Save------------------------------------------

extern int OCFD_ANA_time_average; 
extern int OCFD_ana_flatplate,  OCFD_ana_saveplaneYZ;
extern int OCFD_ana_saveplaneXZ, OCFD_ana_savePoints;
extern int OCFD_ana_saveplaneXY, OCFD_ana_saveblock;
extern int OCFD_ana_getQ;

extern int Kstep_save, Kstep_show,N_ana,*K_ana,*Kstep_ana,KRK;

//------------Scheme invis and vis----------------------------------------
extern int Scheme_invis_ID;
extern int Scheme_vis_ID;

//---------------------WENO_SYMBO_Limiter------------------------------------------
extern REAL WENO_TV_Limiter;
extern REAL WENO_TV_MAX;

// -----------Boundary Condition and Initial Condition-----------------------------
extern int Iperiodic[3], Jacbound[3], D0_bound[6];
extern int Non_ref[6];

extern int Y_Jacbound,Y_bound0;

extern int OCFD_BC_Liftbody3d;
extern int IBC_USER;
extern int Init_stat;

extern char IF_SYMMETRY;
extern char IF_WITHLEADING;  // 0 不含头部， 1 含头部 
extern char IFLAG_UPPERBOUNDARY; // 0 激波外； 1 激波
extern REAL AOA,TW,EPSL_WALL,EPSL_UPPER,WALL_DIS_BEGIN,WALL_DIS_END;
extern REAL Sin_AOA , Cos_AOA;

extern REAL *BC_rpara, (*ANA_rpara)[100]; 
extern int *BC_npara, (*ANA_npara)[100]; 


// ------Filter----------------------------------------------- 
extern int FILTER_1,FILTER_2;   // 1: 5-point filter, 2: 7-point wide-band filter
extern int NF_max;   // maximum of filtering 
extern int Filter_Fo9p, Filter_Fopt_shock;
extern int NFiltering,(*Filter_para)[11];      //Filtering
extern REAL (*Filter_rpara)[3];   // s0  (0<s0<1), rth
extern char IF_Filter_X , IF_Filter_Y , IF_Filter_Z ;
extern int fiter_judge_X, fiter_judge_Y, fiter_judge_Z;
// --------------------------------------------------



// ----------------------------------------------------------
// Coordinate parameters 
extern REAL hx,hy,hz;  //参考空间网格尺寸
extern REAL *pAxx,*pAyy,*pAzz,*pAkx,*pAky,*pAkz,*pAix,*pAiy,*pAiz,*pAsx,*pAsy,*pAsz,*pAjac;  // 度量系数矩阵


// calculate memory
extern REAL *pAmu; // viscous 3d [nz][nt][nx]
extern REAL *pd,*pu,*pv,*pw,*pT,*pP; //  [nz+2*LAP][ny+2*LAP][nx+2*LAP]
extern REAL *pf,*pfn,*pdu; // [nz][ny][nx][5]

// used in filtering
extern REAL *pf_lap; // [nz+2*LAP][ny+2*LAP][nx+2*LAP][5]

extern REAL *pO;
extern REAL *pO2;
extern REAL *pN;
extern REAL *pNO;
extern REAL *pN2;

// used in ana
extern REAL *pdm, *pum, *pvm, *pwm, *pTm;//[nx][ny][nz]
extern int average_IO;
extern int Istep_average;
extern REAL tt_average;

// used in vis jacobian , is part of ptmpb
extern REAL * pEv1,*pEv2,*pEv3,*pEv4;  // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in vis jacobian , is part of ptmpb
extern REAL *puk,*pui,*pus,*pvk,*pvi,*pvs,*pwk,*pwi,*pws,*pTk,*pTi,*pTs;  //[nz][ny][nx]
// used in mecpy
extern REAL *pack_send_x,* pack_recv_x;
extern REAL *pack_send_y,* pack_recv_y;
extern REAL *pack_send_z,* pack_recv_z;
// used in boundary_compressible_conner***********************************************************
extern int MZMAX, MTMAX, INLET_BOUNDARY, IFLAG_WALL_NOT_NORMAL;
extern REAL EPSL, X_DIST_BEGIN, X_DIST_END, BETA;
extern REAL X_WALL_BEGIN, X_UP_BOUNDARY_BEGIN;

// used in SCHEME_HYBRIDAUTO ********************************************************************
extern int IFLAG_HybridAuto;
extern int HybridA_Stage, Patch_max;
extern int IFLAG_mem;
extern HybridAuto_TYPE HybridAuto;
extern int *scheme_x, *scheme_y, *scheme_z;

extern int IF_CHARTERIC;

void read_parameters();

#ifdef __cplusplus
}
#endif
#endif




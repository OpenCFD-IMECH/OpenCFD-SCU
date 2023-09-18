#ifndef __PARAMETERS_D_H
#define __PARAMETERS_D_H

#include "config_parameters.h"
#include "cuda_commen.h"


// -----constant-----
#ifdef __cplusplus
extern "C"{
#endif

extern hipStream_t Stream[15];
extern hipEvent_t  Event[15];

extern __device__ __constant__ REAL Ama_d , Gamma_d , epsl_sw_d;
extern __device__ __constant__ REAL Cv_d , Cp_d , Tsb_d , amu_C0_d;
extern __device__ __constant__ REAL split_C1_d , split_C3_d;
extern __device__ __constant__ REAL vis_flux_init_c_d;

extern __device__ __constant__ unsigned int nx_d,ny_d,nz_d; // 某方向所处理的个数
extern __device__ __constant__ unsigned int nx_lap_d,ny_lap_d,nz_lap_d; // 某方向所处理的个数
extern __device__ __constant__ unsigned int nx_2lap_d,ny_2lap_d,nz_2lap_d; // 某方向所处理的个数

extern __device__ __constant__ REAL dt_d;
extern __device__ __constant__ REAL Sin_AOA_d , Cos_AOA_d;
extern __device__ __constant__ REAL TW_d;

//---------------------WENO_SYMBO_Limiter------------------------------------------
extern __device__ __constant__ REAL WENO_TV_Limiter_d;
extern __device__ __constant__ REAL WENO_TV_MAX_d;

// ----------------------------------------------------------
// Coordinate parameters 
extern __device__ __constant__ REAL hx_d,hy_d,hz_d;  //参考空间网格尺寸

extern  cudaField *pAxx_d,*pAyy_d,*pAzz_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,*pAjac_d;  // 度量系数矩阵


// calculate memory
extern cudaField *pAmu_d; // viscous 3d [nz][nt][nx]
extern cudaField *pd_d,*pu_d,*pv_d,*pw_d,*pT_d,*pP_d; //  [nz+2*LAP][ny+2*LAP][nx+2*LAP]
extern cudaSoA *pf_d,*pfn_d,*pdu_d; // [nz][ny][nx][5]

// used in filtering
extern cudaSoA *pf_lap_d; // [nz+2*LAP][ny+2*LAP][nx+2*LAP][5]

// used in analysis
extern cudaField *pdm_d, *pum_d, *pvm_d, *pwm_d, *pTm_d;//[nz][ny][nx]

// used in invis jacobian , is part of ptmpa
extern cudaSoA *pfp_x_d; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]
extern cudaSoA *pfm_x_d; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]

extern cudaSoA *pfp_y_d; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]
extern cudaSoA *pfm_y_d; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]

extern cudaSoA *pfp_z_d; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]
extern cudaSoA *pfm_z_d; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]

extern cudaField *pcc_d; // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in invis jacobian , is part of ptmpb
extern cudaField *pdfp_d , *pdfm_d; // [nz][ny][nx]



// used in vis jacobian , is part of ptmpb
extern cudaField * pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d;  // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in vis jacobian , is part of ptmpb
extern cudaField *puk_d,*pui_d,*pus_d,*pvk_d,*pvi_d,*pvs_d,*pwk_d,*pwi_d,*pws_d,*pTk_d,*pTi_d,*pTs_d;  //[nz][ny][nx]
extern cudaField *vis_u_d,*vis_v_d,*vis_w_d,*vis_T_d;  //[nz][ny][nx]

extern cudaField grad_P;
extern cudaField *pPP_d;


#ifdef __cplusplus
}
#endif
#endif




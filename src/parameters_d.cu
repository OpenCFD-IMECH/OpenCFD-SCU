#include "config_parameters.h"
/* 
GPU端所需参数与常数
 */

#include "cuda_runtime.h"
#include "cuda_commen.h"
#ifdef __cplusplus
extern "C"{
#endif

cudaStream_t Stream[15];
cudaEvent_t  Event[15];

// -----constant-----
__device__ __constant__ REAL Ama_d , Gamma_d , epsl_sw_d;
__device__ __constant__  REAL Cv_d , Cp_d , Tsb_d , amu_C0_d;

__device__ __constant__ REAL split_C1_d , split_C3_d;
__device__ __constant__ REAL vis_flux_init_c_d;

__device__ __constant__ unsigned int nx_d,ny_d,nz_d; // 某方向所处理的个数
__device__ __constant__ unsigned int nx_lap_d,ny_lap_d,nz_lap_d;
__device__ __constant__ unsigned int nx_2lap_d,ny_2lap_d,nz_2lap_d;
__device__ __constant__ REAL dt_d;
__device__ __constant__ REAL Sin_AOA_d , Cos_AOA_d;
__device__ __constant__ REAL TW_d;

//---------------------WENO_SYMBO_Limiter------------------------------------------
__device__ __constant__ REAL WENO_TV_Limiter_d;
__device__ __constant__ REAL WENO_TV_MAX_d;

// ----------------------------------------------------------
// Coordinate parameters 
__device__ __constant__ REAL hx_d,hy_d,hz_d;  //参考空间网格尺寸
cudaField *pAxx_d,*pAyy_d,*pAzz_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,*pAjac_d;  // 度量系数矩阵


// calculate memory
cudaField *pAmu_d; // viscous 3d [nz][ny][nx]
cudaField *pd_d,*pu_d,*pv_d,*pw_d,*pT_d,*pP_d, *pe_d; //  [nz+2*LAP][ny+2*LAP][nx+2*LAP]
cudaSoA *pf_d,*pfn_d,*pdu_d; // [5][nz][ny][nx]

// memory space for species
cudaSoA *pspec_d, *pspecn_d, *pdspec_d;
cudaField *pO_d,*pO2_d,*pN_d,*pNO_d,*pN2_d; //  [nz+2*LAP][ny+2*LAP][nx+2*LAP]

// used in filtering
cudaSoA *pf_lap_d; // [nz+2*LAP][ny+2*LAP][nx+2*LAP][5]

// used in analysis
cudaField *pdm_d, *pum_d, *pvm_d, *pwm_d, *pTm_d;

// used in invis jacobian , is part of ptmpa
cudaSoA *pfp_x_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]
cudaSoA *pfm_x_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]

cudaSoA *pfp_y_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]
cudaSoA *pfm_y_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]

cudaSoA *pfp_z_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]
cudaSoA *pfm_z_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]

cudaField *pcc_d; // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in invis jacobian , is part of ptmpb
cudaField *pdfp_d , *pdfm_d; // [nz][ny][nx]

cudaSoA *pfpi_x_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]
cudaSoA *pfmi_x_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]

cudaSoA *pfpi_y_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]
cudaSoA *pfmi_y_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]

cudaSoA *pfpi_z_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]
cudaSoA *pfmi_z_d; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP]

// used in vis jacobian , is part of ptmpb
cudaField * pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d;  // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in vis jacobian , is part of ptmpb
cudaField *puk_d,*pui_d,*pus_d,*pvk_d,*pvi_d,*pvs_d,*pwk_d,*pwi_d,*pws_d,*pTk_d,*pTi_d,*pTs_d;  //[nz][ny][nx]
cudaField *vis_u_d,*vis_v_d,*vis_w_d,*vis_T_d;  //[nz][ny][nx]

// used in boundary_liftbody***************************************************


// used in boundary_compressible_conner****************************************
cudaField *pub1_d, *pfx_d, *pgz_d;

cudaField grad_P;
cudaField *pPP_d;

#ifdef __cplusplus
}
#endif

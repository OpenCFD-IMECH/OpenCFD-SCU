#include "math.h"
#include "parameters.h"
#include "utility.h"
#include "OCFD_IO.h"
#include "io_warp.h"
#include "OCFD_Comput_Jacobian3d.h"
#include "OCFD_boundary_init.h"
#include "OCFD_mpi.h"
#include "OCFD_mpi_dev.h"

#include "OCFD_Stream.h"
#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "commen_kernel.h"
#include "OCFD_init.h"
#include "time.h"
#include "mpi.h"
#include "OCFD_ana.h"

#ifdef __cplusplus
extern "C"{
#endif


void init()
{

	hx = 1.0 / (NX_GLOBAL - 1.0);
	hy = 1.0 / (NY_GLOBAL - 1.0);
	hz = 1.0 / (NZ_GLOBAL - 1.0);

    opencfd_para_init();
    opencfd_para_init_dev();
    
	Init_Jacobian3d();

	//----------------------------------------------------------------------------
    {   
        cuda_mem_value_init_warp(0.0 , pd_d->ptr , pd_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pu_d->ptr , pu_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pv_d->ptr , pv_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pw_d->ptr , pw_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pT_d->ptr , pT_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pP_d->ptr , pP_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        //cuda_mem_value_init_warp(0.0 , pf_lap_d->ptr , pf_lap_d->pitch , nx_2lap , ny_2lap , nz_2lap*5);
        cuda_mem_value_init_warp(0.0 , pcc_d->ptr, pcc_d->pitch, nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pf_d->ptr , pf_d->pitch , nx      , ny      , nz*5   );
        cuda_mem_value_init_warp(0.0 , pfn_d->ptr, pfn_d->pitch, nx      , ny      , nz*5   );
        cuda_mem_value_init_warp(0.0 , pdu_d->ptr, pdu_d->pitch, nx      , ny      , nz*5   );
        cuda_mem_value_init_warp(0.0 , pEv1_d->ptr , pEv1_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pEv2_d->ptr , pEv2_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pEv3_d->ptr , pEv3_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(0.0 , pEv4_d->ptr , pEv4_d->pitch , nx_2lap , ny_2lap , nz_2lap);

        REAL * tmp;
        int tmp_size = (nx+2*LAP)*(ny+2*LAP)*(nz+2*LAP);
        tmp = pd; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pu; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pv; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pw; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pT; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pP; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pf_lap; for(int i=0;i<tmp_size*5;i++) (*tmp++) = 0.0;
        tmp = pcc; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pEv1; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pEv2; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pEv3; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pEv4; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;

        tmp_size = nx*ny*nz*5;
        tmp = pf ; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pfn; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
        tmp = pdu; for(int i=0;i<tmp_size;i++) (*tmp++) = 0.0;
    }

	if (Init_stat == 0)
	{
        cuda_mem_value_init_warp(1.0 , pd_d->ptr , pd_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(1.0 , pu_d->ptr , pu_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        cuda_mem_value_init_warp(1.0 , pT_d->ptr , pT_d->pitch , nx_2lap , ny_2lap , nz_2lap);
        
		Istep = 0;
		tt = 0.0;
	}
	else
	{

		read_file(0, pd, pu, pv, pw, pT);

        memcpy_inner(pd , pd_d->ptr , pd_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_inner(pu , pu_d->ptr , pu_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_inner(pv , pv_d->ptr , pv_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_inner(pw , pw_d->ptr , pw_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_inner(pT , pT_d->ptr , pT_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);

	}


	//----------------------------------------------------------------------------
    // pri to cons
    {   
        #ifdef __cplusplus
        cudaJobPackage job(dim3(0,0,0) , dim3(nx,ny,nz));
        #else
        cudaJobPackage job = {0,0,0,nx,ny,nz};
        #endif

        dim3 blockdim = {BlockDimX , BlockDimY , BlockDimZ};
        pri_to_cons_kernel_warp(pf_d , pd_d , pu_d , pv_d , pw_d , pT_d ,job , blockdim);

    }

    //Init_Jacobian3d();
    
    switch(IBC_USER){
        case 124:
        {
            if(Init_stat == 1) bc_user_Liftbody3d_init();
            boundary_Jac3d_Liftbody_Ajac();
        }
        break;

        case 108:
        bc_user_Compression_conner_init();

        break;
    }
}
// ======================================================================= //

void opencfd_mem_init_all(){
    opencfd_mem_init();
    opencfd_mem_init_dev();
//    opencfd_mem_init_dev2();
    opencfd_mem_init_Stream();
    opencfd_mem_init_mpi_dev();
}

void opencfd_mem_finalize_all(){
    opencfd_mem_finalize();
    opencfd_mem_finalize_mpi();
    opencfd_mem_finalize_Stream();
    opencfd_mem_finalize_mpi_dev();
    opencfd_mem_finalize_dev();
    opencfd_mem_finalize_boundary();
}


void opencfd_mem_init()
{
    // space transformation jacobian data
    int tmp_size = (nx + 2 * LAP) * (ny + 2 * LAP) * (nz + 2 * LAP) * sizeof(REAL);

    pAxx = (REAL *)malloc_me(tmp_size);
    pAyy = (REAL *)malloc_me(tmp_size);
    pAzz = (REAL *)malloc_me(tmp_size);
    pAkx = (REAL *)malloc_me(tmp_size);
    pAky = (REAL *)malloc_me(tmp_size);
    pAkz = (REAL *)malloc_me(tmp_size);
    pAix = (REAL *)malloc_me(tmp_size);
    pAiy = (REAL *)malloc_me(tmp_size);
    pAiz = (REAL *)malloc_me(tmp_size);
    pAsx = (REAL *)malloc_me(tmp_size);
    pAsy = (REAL *)malloc_me(tmp_size);
    pAsz = (REAL *)malloc_me(tmp_size);
    pAjac = (REAL *)malloc_me(tmp_size);

    // computing data without LAP
    tmp_size = nx * ny * nz * sizeof(REAL);

    pf  = (REAL *)malloc_me(tmp_size*5);
    pfn = (REAL *)malloc_me(tmp_size*5);
    pdu = (REAL *)malloc_me(tmp_size*5);

    pAmu = (REAL *)malloc_me(tmp_size);

    pdfp = (REAL*)malloc_me(tmp_size*5);
    pdfm = (REAL*)malloc_me(tmp_size*5);

    pui = (REAL*)malloc_me(tmp_size);
    pus = (REAL*)malloc_me(tmp_size);
    puk = (REAL*)malloc_me(tmp_size);
    pvk = (REAL*)malloc_me(tmp_size);
    pvi = (REAL*)malloc_me(tmp_size);
    pvs = (REAL*)malloc_me(tmp_size);
    pwk = (REAL*)malloc_me(tmp_size);
    pwi = (REAL*)malloc_me(tmp_size);
    pws = (REAL*)malloc_me(tmp_size);
    pTk = (REAL*)malloc_me(tmp_size);
    pTi = (REAL*)malloc_me(tmp_size);
    pTs = (REAL*)malloc_me(tmp_size);

    pQ = (REAL*)malloc_me(tmp_size);
    pLamda2 = (REAL*)malloc_me(tmp_size);

    // computing data with LAP
    tmp_size = (nx + 2 * LAP) * (ny + 2 * LAP) * (nz + 2 * LAP) * sizeof(REAL);
    pd = (REAL *)malloc_me(tmp_size);
    pu = (REAL *)malloc_me(tmp_size);
    pv = (REAL *)malloc_me(tmp_size);
    pw = (REAL *)malloc_me(tmp_size);
    pT = (REAL *)malloc_me(tmp_size);
    pP = (REAL *)malloc_me(tmp_size);

    pf_lap = (REAL*)malloc_me(tmp_size*5);
    pfp = (REAL*)malloc_me(tmp_size*5);
    pfm = (REAL*)malloc_me(tmp_size*5);
    pcc = (REAL*)malloc_me(tmp_size);

    pEv1 = (REAL*)malloc_me(tmp_size);
    pEv2 = (REAL*)malloc_me(tmp_size);
    pEv3 = (REAL*)malloc_me(tmp_size);
    pEv4 = (REAL*)malloc_me(tmp_size);

    tmp_size = LAP * ny * nz * sizeof(REAL);
    malloc_me_Host(pack_send_x, tmp_size);
    malloc_me_Host(pack_recv_x, tmp_size);

    tmp_size = LAP * nx * nz * sizeof(REAL);
    malloc_me_Host(pack_send_y, tmp_size);
    malloc_me_Host(pack_recv_y, tmp_size);

    tmp_size = LAP * nx * ny * sizeof(REAL);
    malloc_me_Host(pack_send_z, tmp_size);
    malloc_me_Host(pack_recv_z, tmp_size);

    if(IFLAG_HybridAuto == 1){
        scheme_x = (int *)malloc_me((nx + 1)*ny*nz*sizeof(int));
        scheme_y = (int *)malloc_me((ny + 1)*nx*nz*sizeof(int));
        scheme_z = (int *)malloc_me((nz + 1)*nx*ny*sizeof(int));
    }
}

void opencfd_mem_finalize()
{

    free(pAxx);
    free(pAyy);
    free(pAzz);
    free(pAkx);
    free(pAky);
    free(pAkz);
    free(pAix);
    free(pAiy);
    free(pAiz);
    free(pAsx);
    free(pAsy);
    free(pAsz);
    free(pAjac);

    
    free(pf);
    free(pfn);
    free(pdu);

    free(pAmu);

    free(pdfp);
    free(pdfm);

    free(pui);
    free(pus);
    free(puk);
    free(pvk);
    free(pvi);
    free(pvs);
    free(pwk);
    free(pwi);
    free(pws);
    free(pTk);
    free(pTi);
    free(pTs);

    free(pd);
    free(pu);
    free(pv);
    free(pw);
    free(pT);
    free(pP);
    
    free(pf_lap);
    free(pfp);
    free(pfm);
    free(pcc);
    
    free(pEv1);
    free(pEv2);
    free(pEv3);
    free(pEv4);
    free(pQ);
    free(pLamda2);

    hipHostFree(pack_send_x);
    hipHostFree(pack_recv_x);

    hipHostFree(pack_send_y);
    hipHostFree(pack_recv_y);

    hipHostFree(pack_send_z);
    hipHostFree(pack_recv_z);

}

void opencfd_mem_init_boundary(){
    switch(IBC_USER){
        case 124:

        pu2d_inlet = (REAL*)malloc_me(sizeof(REAL)*5*nz*ny);
        pu2d_upper = (REAL*)malloc_me(sizeof(REAL)*5*ny*nx);
        pv_dist_wall = (REAL*)malloc_me(sizeof(REAL)*ny*nx);
        pv_dist_coeff = (REAL*)malloc_me(sizeof(REAL)*3*ny*nx);
        pu_dist_upper = (REAL*)malloc_me(sizeof(REAL)*ny*nx);
        pfx = (REAL*)malloc_me(sizeof(REAL)*nx);
        pgz = (REAL*)malloc_me(sizeof(REAL)*ny);
        TM = (REAL*)malloc_me(sizeof(REAL)*MTMAX);
        fait = (REAL*)malloc_me(sizeof(REAL)*MTMAX);
        
        new_cudaField(&pu2d_inlet_d, ny, nz, 5);
        new_cudaField(&pu2d_upper_d, nx, ny, 5);
        //new_cudaField(&pv_dist_wall_d , nx,ny,1);
        new_cudaField(&pv_dist_coeff_d, nx, ny, 3);
        new_cudaField(&pu_dist_upper_d, nx, ny, 1);
        break;
    
        case 108:
        pub1 = (REAL*)malloc_me(sizeof(REAL)*4*ny);
        pfx = (REAL*)malloc_me(sizeof(REAL)*nx);
        pgz = (REAL*)malloc_me(sizeof(REAL)*nz);
    
        TM = (REAL*)malloc_me(sizeof(REAL)*MTMAX);
        fait = (REAL*)malloc_me(sizeof(REAL)*MTMAX);
        //ptmp = (REAL*)malloc_me_Host(sizeof(REAL)*10*nx*nz);
    
        new_cudaField(&pub1_d, ny, 4, 1);
        new_cudaField(&pfx_d, nx, 1, 1);
        new_cudaField(&pgz_d, nz, 1, 1);
        //new_cudaField(&ptmp_d, nx, nz, 10);
        break;
    }
}

void opencfd_mem_finalize_boundary(){
    switch(IBC_USER){
        case 124:
        free(pu2d_inlet);
	    free(pu2d_upper);
	    free(pv_dist_wall);
	    free(pv_dist_coeff);
	    free(pu_dist_upper);
        free(pfx);
        free(pgz);

        free(TM);
        free(fait);
    
        delete_cudaField(pu2d_inlet_d);
	    delete_cudaField(pu2d_upper_d);
	    //delete_cudaField(pv_dist_wall_d);
	    delete_cudaField(pv_dist_coeff_d);
	    //delete_cudaField(pu_dist_upper_d);
        break;

        case 108:
        free(pub1);
        free(pfx);
        free(pgz);

        free(TM);
        free(fait);

        delete_cudaField(pub1_d);
        delete_cudaField(pfx_d);
        delete_cudaField(pgz_d);
        break;
    }
}

void opencfd_para_init(){
    Cv = 1.0 / (Gamma * (Gamma - 1.0) * Ama * Ama);
    Cp = 1.0 / ((Gamma - 1.0) * Ama * Ama);
    Tsb = 110.4 / Ref_T;

    amu_C0 = 1.0 / Re * (1.0 + 110.4 / Ref_T);
    tmp0 = 1.0 / (2.0 * Gamma);
    split_C1 = 2.0 * (Gamma - 1.0); 
    split_C3 = (3.0 - Gamma) / (2.0 * (Gamma - 1.0));

    nx_lap = nx+LAP;
    ny_lap = ny+LAP;
    nz_lap = nz+LAP;

    nx_2lap = nx + 2*LAP;
    ny_2lap = ny + 2*LAP;
    nz_2lap = nz + 2*LAP;
    
    vis_flux_init_c = Cp/Pr;

    end_step = (int)ceil(end_time/dt);

    // 
    int flag;
	MPI_Initialized(&flag);
	if(!flag){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if(my_id == 0) printf("%s finished\n",__func__);
}

// ======================================================================================== //

void opencfd_mem_init_dev(){
    new_cudaField(&pAxx_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAyy_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAzz_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAkx_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAky_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAkz_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAix_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAiy_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAiz_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAsx_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAsy_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAsz_d  , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pAjac_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);

    new_cudaField(&pAmu_d , nx,ny,nz);
    new_cudaField(&pd_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pu_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pv_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pw_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pT_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField(&pP_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);

    new_cudaSoA(&pf_d  , nx , ny , nz);
    new_cudaSoA(&pfn_d , nx , ny , nz);
    new_cudaSoA(&pdu_d , nx , ny , nz);

    new_cudaField( &pcc_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);

    new_cudaSoA( &pfp_x_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaSoA( &pfm_x_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);

    pf_lap_d = (cudaSoA *)malloc(sizeof(cudaSoA));
    pf_lap_d->ptr = pfp_x_d->ptr;
    pf_lap_d->pitch = pfp_x_d->pitch;

    new_cudaSoA( &pfp_y_d , nx+2*LAP, ny+2*LAP, nz+2*LAP);
    new_cudaSoA( &pfm_y_d , nx+2*LAP, ny+2*LAP, nz+2*LAP);

    new_cudaSoA( &pfp_z_d , nx+2*LAP, ny+2*LAP, nz+2*LAP);
    new_cudaSoA( &pfm_z_d , nx+2*LAP, ny+2*LAP, nz+2*LAP);

    new_cudaField( &vis_u_d , nx , ny , nz);
    new_cudaField( &vis_v_d , nx , ny , nz);
    new_cudaField( &vis_w_d , nx , ny , nz);
    new_cudaField( &vis_T_d , nx , ny , nz);

    new_cudaField( &pEv1_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField( &pEv2_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField( &pEv3_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    new_cudaField( &pEv4_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);

    new_cudaField( &puk_d , nx , ny , nz);
    new_cudaField( &pui_d , nx , ny , nz);
    new_cudaField( &pus_d , nx , ny , nz);

    new_cudaField( &pvk_d , nx , ny , nz);
    new_cudaField( &pvi_d , nx , ny , nz);
    new_cudaField( &pvs_d , nx , ny , nz);

    new_cudaField( &pwk_d , nx , ny , nz);
    new_cudaField( &pwi_d , nx , ny , nz);
    new_cudaField( &pws_d , nx , ny , nz);

    new_cudaField( &pTk_d , nx , ny , nz);
    new_cudaField( &pTi_d , nx , ny , nz);
    new_cudaField( &pTs_d , nx , ny , nz);

    if(IFLAG_HybridAuto == 1){
        new_cudaField_int(&(HybridAuto.scheme_x), nx+1, ny, nz);
        new_cudaField_int(&(HybridAuto.scheme_y), nx, ny+1, nz);
        new_cudaField_int(&(HybridAuto.scheme_z), nx, ny, nz+1);
        new_cudaField(&pPP_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
    }
}


void opencfd_mem_finalize_dev(){
    delete_cudaField(pAxx_d  );
    delete_cudaField(pAyy_d  );
    delete_cudaField(pAzz_d  );
    delete_cudaField(pAkx_d  );
    delete_cudaField(pAky_d  );
    delete_cudaField(pAkz_d  );
    delete_cudaField(pAix_d  );
    delete_cudaField(pAiy_d  );
    delete_cudaField(pAiz_d  );
    delete_cudaField(pAsx_d  );
    delete_cudaField(pAsy_d  );
    delete_cudaField(pAsz_d  );
    delete_cudaField(pAjac_d );

    delete_cudaField(pAmu_d);

    delete_cudaField(pd_d);
    delete_cudaField(pu_d);
    delete_cudaField(pv_d);
    delete_cudaField(pw_d);
    delete_cudaField(pT_d);
    delete_cudaField(pP_d);

    delete_cudaSoA(pf_d );
    delete_cudaSoA(pfn_d);
    delete_cudaSoA(pdu_d);

    //delete_cudaSoA(pf_lap_d);

    delete_cudaSoA(pfp_x_d);
    delete_cudaSoA(pfm_x_d);

    delete_cudaSoA(pfp_y_d);
    delete_cudaSoA(pfm_y_d);

    delete_cudaSoA(pfp_z_d);
    delete_cudaSoA(pfm_z_d);
    
    delete_cudaField(pcc_d);
    
    //delete_cudaField(pdfp_d);
    //delete_cudaField(pdfm_d);

    delete_cudaField(vis_u_d);
    delete_cudaField(vis_v_d);
    delete_cudaField(vis_w_d);
    delete_cudaField(vis_T_d);

    delete_cudaField(pEv1_d);
    delete_cudaField(pEv2_d);
    delete_cudaField(pEv3_d);
    delete_cudaField(pEv4_d);

    delete_cudaField( puk_d );
    delete_cudaField( pui_d );
    delete_cudaField( pus_d );

    delete_cudaField( pvk_d);
    delete_cudaField( pvi_d);
    delete_cudaField( pvs_d);

    delete_cudaField( pwk_d );
    delete_cudaField( pwi_d );
    delete_cudaField( pws_d );

    delete_cudaField( pTk_d );
    delete_cudaField( pTi_d );
    delete_cudaField( pTs_d );
    
}


#ifndef __NVCC__

#ifndef HIP_SYMBOL
#define HIP_SYMBOL( var ) (&var)
#endif

#else

#ifndef HIP_SYMBOL
#define HIP_SYMBOL( var ) (var)
#endif

#endif



void opencfd_para_init_dev(){
    hipMemcpyToSymbol( HIP_SYMBOL(Ama_d) , &Ama , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(Gamma_d) , &Gamma , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(epsl_sw_d) , &epsl_SW , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(Cv_d) , &Cv , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(Cp_d) , &Cp , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(Tsb_d) , &Tsb , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(amu_C0_d) , &amu_C0 , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(dt_d) , &dt , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(split_C1_d) , &split_C1 , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(split_C3_d) , &split_C3 , sizeof(REAL) , 0 , hipMemcpyHostToDevice);


    hipMemcpyToSymbol( HIP_SYMBOL(vis_flux_init_c_d) , &vis_flux_init_c , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(nx_d) , &nx , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(ny_d) , &ny , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(nz_d) , &nz , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(nx_lap_d) , &nx_lap , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(ny_lap_d) , &ny_lap , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(nz_lap_d) , &nz_lap , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(nx_2lap_d) , &nx_2lap , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(ny_2lap_d) , &ny_2lap , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(nz_2lap_d) , &nz_2lap , sizeof(unsigned int) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(hx_d) , &hx , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(hy_d) , &hy , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(hz_d) , &hz , sizeof(REAL) , 0 , hipMemcpyHostToDevice);

    hipMemcpyToSymbol( HIP_SYMBOL(WENO_TV_Limiter_d) , &WENO_TV_Limiter , sizeof(REAL) , 0 , hipMemcpyHostToDevice);
    hipMemcpyToSymbol( HIP_SYMBOL(WENO_TV_MAX_d) , &WENO_TV_MAX , sizeof(REAL) , 0 , hipMemcpyHostToDevice);

    int flag;
	MPI_Initialized(&flag);
	if(!flag){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if(my_id == 0) printf("%s finished\n",__func__);
}

#ifdef __cplusplus
}
#endif

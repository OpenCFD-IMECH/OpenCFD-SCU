#include <math.h>
#include "parameters.h"
#include "utility.h"
#include "OCFD_Schemes.h"
#include "OCFD_Schemes_Choose.h"
#include "OCFD_Schemes_hybrid_auto.h"
#include "OCFD_bound_Scheme.h"
#include "OCFD_flux_charteric.h"

#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"

#ifdef __cplusplus 
extern "C"{
#endif

// Used in viscous Jacobian --------------------------------------------------------------------------------------------
void OCFD_dx0(cudaField pf, cudaField pfx, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;

	dim3 flagxyzb(1, 0, 0);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	cal_grid_block_dim(&griddim, &blockdim, 8, 8, 4, size.x, size.y, size.z);

	switch(Scheme_vis_ID){
		case 203:
		CUDA_LAUNCH((OCFD_CD6_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pfx, job_in) ));
		break;

		case 204:
		CUDA_LAUNCH((OCFD_CD8_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pfx, job_in) ));
		break;
	}

}


void OCFD_dy0(cudaField pf, cudaField pfy, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;

	dim3 flagxyzb(2, 0, 0);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.x, blockdim_in.y, blockdim_in.z, size.x, size.y, size.z );

	switch(Scheme_vis_ID){
		case 203:
		CUDA_LAUNCH((OCFD_CD6_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pfy, job_in) ));
		break;

		case 204:
		CUDA_LAUNCH((OCFD_CD8_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pfy, job_in) ));
		break;
	}

}

void OCFD_dz0(cudaField pf, cudaField pfz, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;

	dim3 flagxyzb(3, 0, 0);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.x, blockdim_in.y, blockdim_in.z, size.x, size.z, size.y );

	switch(Scheme_vis_ID){
		case 203:
		CUDA_LAUNCH((OCFD_CD6_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pfz, job_in) ));
		break;

		case 204:
		CUDA_LAUNCH((OCFD_CD8_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pfz, job_in) ));
		break;
	}

}


// Used in inviscous Jacobian flux+ ------------------------------------------------------------------------------------------
void OCFD_dx1(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
// field with LAPs
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.x-1, blockdim_in.y, blockdim_in.z, size.x, size.y, size.z );

	dim3 flagxyzb(1, 0, Non_ref[0]);//.x正向边界；.y负向边界；.z无反射边界
    OCFD_bound_non_ref(&flagxyzb, Non_ref[0], job_in);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	job_in.start.x  -= 1;
    blockdim.x += 1;

	if(IF_CHARTERIC == 1){
	    switch(Scheme_invis_ID){
	    	case 301:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 302:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 303:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_P_kernel<<<griddim, blockdim, 0, *stream>>>(0, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_P_kernel<<<griddim, blockdim, 0, *stream>>>(1, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 307:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 308:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 309:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_P_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_x, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_P_Jameson_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_x, job_in) ));
			}
	    	break;
	    }
	}else{
        for(int i=0; i<5; i++){
		switch(Scheme_invis_ID){
	    	case 301:
	    	CUDA_LAUNCH((OCFD_UP7_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 302:
	    	CUDA_LAUNCH((OCFD_weno5_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 303:
	    	CUDA_LAUNCH((OCFD_weno7_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH(( OCFD_weno7_SYMBO_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
	    	CUDA_LAUNCH((OCFD_NND2_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 307:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 308:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 309:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, 2, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_P_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_x, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_P_Jameson_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_x, job_in) ));
			}
			break;
	    }
        }
	}

}


void OCFD_dy1(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){

	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, 8, 7, 4, size.x, size.y, size.z);

	dim3 flagxyzb(2, 0, Non_ref[2]);
    OCFD_bound_non_ref(&flagxyzb, Non_ref[2], job_in);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

    job_in.start.y  -= 1;
    blockdim.y += 1;

	if(IF_CHARTERIC == 1){
	    switch(Scheme_invis_ID){
	    	case 301:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 302:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 303:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(0, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(1, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 307:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 308:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 309:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_y, job_in) ));
			}else if(HybridAuto.Style == 2){
				CUDA_LAUNCH((OCFD_HybridAuto_character_P_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_y, job_in) ));
			}
			break;
	    }
	}else{
        for(int i=0; i<5; i++){
		switch(Scheme_invis_ID){
	    	case 301:
	    	CUDA_LAUNCH((OCFD_UP7_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 302:
	    	CUDA_LAUNCH((OCFD_weno5_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 303:
	    	CUDA_LAUNCH((OCFD_weno7_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH(( OCFD_weno7_SYMBO_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
	    	CUDA_LAUNCH((OCFD_NND2_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 307:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 308:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 309:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 2, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_y, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_P_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_y, job_in) ));
			}
			break;
	    }
        }
	}
}


// Used in inviscous Jacobian flux- ------------------------------------------------------------------------------------------
void OCFD_dz1(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, 8, 7, 4, size.x, size.z, size.y);

	dim3 flagxyzb(3, 0, Non_ref[4]);
    OCFD_bound_non_ref(&flagxyzb, Non_ref[4], job_in);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

    job_in.start.z  -= 1;
    blockdim.y += 1;

	if(IF_CHARTERIC == 1){
	    switch(Scheme_invis_ID){
	    	case 301:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 302:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 303:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(0, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(1, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 307:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 308:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 309:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_z, job_in) ));
	    	}else if(HybridAuto.Style == 2){
				CUDA_LAUNCH((OCFD_HybridAuto_character_P_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_z, job_in) ));
			}
			break;
	    }
	}else{
        for(int i=0; i<5; i++){
		switch(Scheme_invis_ID){
	    	case 301:
	    	CUDA_LAUNCH((OCFD_UP7_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 302:
	    	CUDA_LAUNCH((OCFD_weno5_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;

	    	case 303:
	    	CUDA_LAUNCH((OCFD_weno7_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH(( OCFD_weno7_SYMBO_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
	    	CUDA_LAUNCH((OCFD_NND2_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 307:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 308:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 309:
	    	CUDA_LAUNCH((OCFD_OMP6_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 2, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_P_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_z, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_P_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_z, job_in) ));
			}
			break;
	    }
        }
	}
}


void OCFD_dx2(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
// field with LAPs
		
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.x-1, blockdim_in.y, blockdim_in.z, size.x, size.y, size.z );

	dim3 flagxyzb(4, 0, Non_ref[1]);
    OCFD_bound_non_ref(&flagxyzb, Non_ref[1], job_in);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	job_in.end.x  += 1;
    blockdim.x += 1;

	if(IF_CHARTERIC == 1){
	    switch(Scheme_invis_ID){
	    	case 301:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 302:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 303:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_M_kernel<<<griddim, blockdim, 0, *stream>>>(0, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_M_kernel<<<griddim, blockdim, 0, *stream>>>(1, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 307:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 308:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 309:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_M_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_x, job_in) ));
			}else if(HybridAuto.Style == 2){
				CUDA_LAUNCH((OCFD_HybridAuto_character_M_Jameson_kernel<<<griddim, blockdim, 0, *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_x, job_in) ));
			}
			break;
	    }
	}else{
        for(int i=0; i<5; i++){
		switch(Scheme_invis_ID){
	    	case 301:
	    	CUDA_LAUNCH((OCFD_UP7_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 302:
	    	CUDA_LAUNCH((OCFD_weno5_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 303:
	    	CUDA_LAUNCH((OCFD_weno7_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
	    	CUDA_LAUNCH((OCFD_NND2_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 307:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 308:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 309:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, 2, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_M_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_x, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_M_Jameson_kernel<<<griddim, blockdim, 0, *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_x, job_in) ));
			}
			break;
	    }
        }
	}
}


void OCFD_dy2(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){

	dim3 size;
	jobsize(&job_in , &size);
	dim3 griddim , blockdim;
	cal_grid_block_dim(&griddim, &blockdim, 8, 7, 4, size.x, size.y, size.z);

	dim3 flagxyzb(5, 0, Non_ref[3]);
    OCFD_bound_non_ref(&flagxyzb, Non_ref[3], job_in);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	job_in.end.y  += 1;
    blockdim.y += 1;

	if(IF_CHARTERIC == 1){
	    switch(Scheme_invis_ID){
	    	case 301:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 302:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 303:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(0, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(1, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 307:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 308:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 309:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_y, job_in) ));
			}else if(HybridAuto.Style == 2){
				CUDA_LAUNCH((OCFD_HybridAuto_character_M_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_y, job_in) ));
			}
	    	break;
	    }
	}else{
        for(int i=0; i<5; i++){
		switch(Scheme_invis_ID){
	    	case 301:
	    	CUDA_LAUNCH((OCFD_UP7_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 302:
	    	CUDA_LAUNCH((OCFD_weno5_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 303:
	    	CUDA_LAUNCH((OCFD_weno7_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
	    	CUDA_LAUNCH((OCFD_NND2_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 307:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 308:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 309:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 2, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_y, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_M_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_y, job_in) ));
			}
			break;
	    }
        }
	}
}


void OCFD_dz2(cudaSoA pf, cudaSoA pdu, cudaField Ajac, cudaField u, cudaField v, cudaField w, cudaField cc, 
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int boundl, int boundr){
	dim3 size;
	jobsize(&job_in , &size);
	dim3 griddim , blockdim;
	cal_grid_block_dim(&griddim, &blockdim, 8, 7, 4, size.x, size.z, size.y);

	dim3 flagxyzb(6, 0, Non_ref[5]);
    OCFD_bound_non_ref(&flagxyzb, Non_ref[5], job_in);
	OCFD_bound(&flagxyzb, boundl, boundr, job_in);

	job_in.end.z  += 1;
    blockdim.y += 1;

	if(IF_CHARTERIC == 1){
	    switch(Scheme_invis_ID){
	    	case 301:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 302:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;

	    	case 303:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(0, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_character_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(1, flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 307:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 308:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 309:
            if(my_id == 0) printf("This scheme does not support charteric flux reconstruction\n");
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_character_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_z, job_in) ));
			}else if(HybridAuto.Style == 2){
				CUDA_LAUNCH((OCFD_HybridAuto_character_M_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(flagxyzb, pf, pdu, Ajac, u, v, w, cc, Ax, Ay, Az, *HybridAuto.scheme_z, job_in) ));
			}
			break;
	    }
	}else{
        for(int i=0; i<5; i++){
		switch(Scheme_invis_ID){
	    	case 301:
	    	CUDA_LAUNCH((OCFD_UP7_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 302:
	    	CUDA_LAUNCH((OCFD_weno5_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;

	    	case 303:
	    	CUDA_LAUNCH((OCFD_weno7_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 304:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo
	    	break;
    
	    	case 305:
	    	CUDA_LAUNCH((OCFD_weno7_SYMBO_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));//weno7_symbo_limiter
	    	break;
    
	    	case 306:
	    	CUDA_LAUNCH((OCFD_NND2_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 307:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 0, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 308:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 1, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 309:
	    	CUDA_LAUNCH((OCFD_OMP6_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, 2, flagxyzb, pf, pdu, Ajac, job_in) ));
	    	break;
    
	    	case 310:
			if(HybridAuto.Style == 1){
	    		CUDA_LAUNCH((OCFD_HybridAuto_M_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_z, job_in) ));
			}else if(HybridAuto.Style == 2){
	    		CUDA_LAUNCH((OCFD_HybridAuto_M_Jameson_kernel<<<griddim, blockdim, 16*8*4*sizeof(REAL), *stream>>>(i, flagxyzb, pf, pdu, Ajac, *HybridAuto.scheme_z, job_in) ));
			}
			break;
	    }
        }
	}
}


void OCFD_dx0_jac(cudaField pf, cudaField pfx, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int bound){
// field with LAPs
	
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.x, blockdim_in.y, blockdim_in.z, size.x, size.y, size.z);
	
	CUDA_LAUNCH(( OCFD_dx0_CD6_kernel<<<griddim, blockdim, 0, *stream>>>(pf, pfx, job_in) ));

	if(bound != 0){
		OCFD_Dx0_bound(pf, pfx, job_in, blockdim_in, stream);
	}
}
	
void OCFD_dy0_jac(cudaField pf, cudaField pfy, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int bound){
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.y, blockdim_in.x, blockdim_in.z, size.y, size.x, size.z );

	CUDA_LAUNCH(( OCFD_dy0_CD6_kernel<<<griddim, blockdim, 0, *stream>>>(pf, pfy, job_in) ));
	
	if(bound != 0){
		OCFD_Dy0_bound(pf, pfy, job_in, blockdim_in, stream);
	}
}
	
void OCFD_dz0_jac(cudaField pf, cudaField pfz, cudaJobPackage job_in, dim3 blockdim_in, cudaStream_t *stream, int bound){
	dim3 size;
	jobsize(&job_in, &size);
	dim3 griddim, blockdim;
	cal_grid_block_dim(&griddim, &blockdim, blockdim_in.z, blockdim_in.x, blockdim_in.y, size.z, size.x, size.y );
	
	CUDA_LAUNCH(( OCFD_dz0_CD6_kernel<<<griddim, blockdim, 0, *stream>>>(pf, pfz, job_in) ));
	
	if(bound != 0){
		OCFD_Dz0_bound(pf, pfz, job_in, blockdim_in, stream);
	}
}

#ifdef __cplusplus
}
#endif

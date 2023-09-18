// =============================================================================================
//  含三维Jocabian变换

#include <math.h>
#include "utility.h"
#include "OCFD_split.h"
#include "parameters.h"
#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"

#ifdef __cplusplus
extern "C"{
#endif

/*
__global__ void split_Jac3d_Stager_Warming_ker(cudaField d0, cudaField u0, cudaField v0, cudaField w0, cudaField cc0, cudaSoA fp, cudaSoA fm, cudaField Akx, cudaField Aky, cudaField Akz, REAL tmp0, REAL split_C1, REAL split_C3, cudaJobPackage job)
{
	// eyes on cells WITH LAPs

	unsigned int x = threadIdx.x + blockIdx.x*blockDim.x + job.start.x;
	unsigned int y = threadIdx.y + blockIdx.y*blockDim.y + job.start.y;
	unsigned int z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
        REAL A1, A2, A3, ss;
	    REAL E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
	    REAL uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;

        REAL u, v, w, cc, d;

		A1 = get_Field_LAP(Ax, x, y, z);
		A2 = get_Field_LAP(Ay, x, y, z);
		A3 = get_Field_LAP(Az, x, y, z);

		ss = sqrt(A1*A1 + A2*A2 + A3*A3);

        d = get_Field_LAP(d0, x, y, z);
        u = get_Field_LAP(u0, x, y, z);
        v = get_Field_LAP(v0, x, y, z);
        w = get_Field_LAP(w0, x, y, z);
        cc = get_Field_LAP(cc0, x, y, z);

		E1 = A1*u + A2*v + A3*w;
		E2 = E1 - cc*ss;
		E3 = E1 + cc*ss;

        ss = 1.0/ss;

        A1 *= ss;
		A2 *= ss;
		A3 *= ss;

        tmp0 = d*tmp0;

		E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
		E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
		E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

		E1M = E1 - E1P;
		E2M = E2 - E2P;
		E3M = E3 - E3P;
		// ----------------------------------------

		uc1 = u - cc * A1;
		uc2 = u + cc * A1;
		vc1 = v - cc * A2;
		vc2 = v + cc * A2;
		wc1 = w - cc * A3;
		wc2 = w + cc * A3;
		vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
		vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50;
		vv = (Gamma_d - 1.0) * (u * u + v * v + w * w );
		W2 = split_C3 * cc * cc;

		
		get_SoA_LAP(fp, x, y, z, 0) = tmp0 * (split_C1 * E1P + E2P + E3P);
		get_SoA_LAP(fp, x, y, z, 1) = tmp0 * (split_C1 * E1P * u + E2P * uc1 + E3P * uc2);
		get_SoA_LAP(fp, x, y, z, 2) = tmp0 * (split_C1 * E1P * v + E2P * vc1 + E3P * vc2);
		get_SoA_LAP(fp, x, y, z, 3) = tmp0 * (split_C1 * E1P * w + E2P * wc1 + E3P * wc2);
		get_SoA_LAP(fp, x, y, z, 4) = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P));
		// --------------------------------------------------------

		get_SoA_LAP(fm, x, y, z, 0) = tmp0 * (split_C1 * E1M + E2M + E3M);
		get_SoA_LAP(fm, x, y, z, 1) = tmp0 * (split_C1 * E1M * u + E2M * uc1 + E3M * uc2);
		get_SoA_LAP(fm, x, y, z, 2) = tmp0 * (split_C1 * E1M * v + E2M * vc1 + E3M * vc2);
		get_SoA_LAP(fm, x, y, z, 3) = tmp0 * (split_C1 * E1M * w + E2M * wc1 + E3M * wc2);
		get_SoA_LAP(fm, x, y, z, 4) = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
	}
}*/

__device__ void Stager_Warming_ker(unsigned int x, unsigned int y, unsigned int z, REAL tmp0, REAL u, REAL v, REAL w, REAL cc, cudaSoA fp, cudaSoA fm, REAL A1, REAL A2, REAL A3, REAL split_C1, REAL split_C3){
        REAL ss;
	    REAL E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
	    REAL uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;

		ss = sqrt(A1*A1 + A2*A2 + A3*A3);

		E1 = A1*u + A2*v + A3*w;
		E2 = E1 - cc*ss;
		E3 = E1 + cc*ss;

        ss = 1.0/ss;

        A1 *= ss;
		A2 *= ss;
		A3 *= ss;

		E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
		E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
		E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

		E1M = E1 - E1P;
		E2M = E2 - E2P;
		E3M = E3 - E3P;
		// ----------------------------------------

		uc1 = u - cc * A1;
		uc2 = u + cc * A1;
		vc1 = v - cc * A2;
		vc2 = v + cc * A2;
		wc1 = w - cc * A3;
		wc2 = w + cc * A3;
		vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
		vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50;
		vv = (Gamma_d - 1.0) * (u * u + v * v + w * w );
		W2 = split_C3 * cc * cc;

		
		get_SoA_LAP(fp, x, y, z, 0) = tmp0 * (split_C1 * E1P + E2P + E3P);
		get_SoA_LAP(fp, x, y, z, 1) = tmp0 * (split_C1 * E1P * u + E2P * uc1 + E3P * uc2);
		get_SoA_LAP(fp, x, y, z, 2) = tmp0 * (split_C1 * E1P * v + E2P * vc1 + E3P * vc2);
		get_SoA_LAP(fp, x, y, z, 3) = tmp0 * (split_C1 * E1P * w + E2P * wc1 + E3P * wc2);
		get_SoA_LAP(fp, x, y, z, 4) = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P));
		// --------------------------------------------------------

		get_SoA_LAP(fm, x, y, z, 0) = tmp0 * (split_C1 * E1M + E2M + E3M);
		get_SoA_LAP(fm, x, y, z, 1) = tmp0 * (split_C1 * E1M * u + E2M * uc1 + E3M * uc2);
		get_SoA_LAP(fm, x, y, z, 2) = tmp0 * (split_C1 * E1M * v + E2M * vc1 + E3M * vc2);
		get_SoA_LAP(fm, x, y, z, 3) = tmp0 * (split_C1 * E1M * w + E2M * wc1 + E3M * wc2);
		get_SoA_LAP(fm, x, y, z, 4) = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
}


__global__ void split_Jac3d_Stager_Warming_ker(sw_split sw, cudaSoA fp_x, cudaSoA fm_x, cudaSoA fp_y, cudaSoA fm_y, cudaSoA fp_z, cudaSoA fm_z, REAL tmp0, REAL split_C1, REAL split_C3, cudaJobPackage job)
{
	// eyes on cells WITH LAPs

	unsigned int x = threadIdx.x + blockIdx.x*blockDim.x + job.start.x;
	unsigned int y = threadIdx.y + blockIdx.y*blockDim.y + job.start.y;
	unsigned int z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
        REAL A1, A2, A3;

        REAL u, v, w, cc, d;

        d = get_Field_LAP(sw.d, x, y, z);
        u = get_Field_LAP(sw.u, x, y, z);
        v = get_Field_LAP(sw.v, x, y, z);
        w = get_Field_LAP(sw.w, x, y, z);
        cc = get_Field_LAP(sw.cc, x, y, z);

        tmp0 = d*tmp0;

		A1 = get_Field_LAP(sw.Akx, x, y, z);
		A2 = get_Field_LAP(sw.Aky, x, y, z);
		A3 = get_Field_LAP(sw.Akz, x, y, z); 

        Stager_Warming_ker(x, y, z, tmp0, u, v, w, cc, fp_x, fm_x, A1, A2, A3, split_C1, split_C3);

        A1 = get_Field_LAP(sw.Aix, x, y, z);
		A2 = get_Field_LAP(sw.Aiy, x, y, z);
		A3 = get_Field_LAP(sw.Aiz, x, y, z); 

        Stager_Warming_ker(x, y, z, tmp0, u, v, w, cc, fp_y, fm_y, A1, A2, A3, split_C1, split_C3);

        A1 = get_Field_LAP(sw.Asx, x, y, z);
		A2 = get_Field_LAP(sw.Asy, x, y, z);
		A3 = get_Field_LAP(sw.Asz, x, y, z); 

        Stager_Warming_ker(x, y, z, tmp0, u, v, w, cc, fp_z, fm_z, A1, A2, A3, split_C1, split_C3);
	}
}

void Stager_Warming(cudaJobPackage job_in, cudaSoA *fp_x, cudaSoA *fm_x, cudaSoA *fp_y, cudaSoA *fm_y, cudaSoA *fp_z, cudaSoA *fm_z, cudaStream_t *stream){
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	//cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x+2*LAP, size.y+2*LAP, size.z+2*LAP);
	cal_grid_block_dim(&griddim, &blockdim, 8, 4, 4, size.x+2*LAP, size.y+2*LAP, size.z+2*LAP);

	cudaJobPackage job( dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP), 
	                    dim3(job_in.end.x + LAP, job_in.end.y + LAP, job_in.end.z + LAP) );

    sw_split sw = {*pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAkx_d, *pAky_d, *pAkz_d, *pAix_d, *pAiy_d, *pAiz_d, *pAsx_d, *pAsy_d, *pAsz_d};
	
	CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker<<<griddim , blockdim, 0, *stream>>>(sw, *fp_x, *fm_x, *fp_y, *fm_y, *fp_z, *fm_z, tmp0, split_C1, split_C3, job) ));
}

/*
__global__ void split_Jac3d_Stager_Warming_ker_out(sw_split_out sw, cudaSoA fp, cudaSoA fm, REAL tmp0, REAL split_C1, REAL split_C3, cudaJobPackage job)
{
	// eyes on cells WITH LAPs

	unsigned int x = threadIdx.x + blockIdx.x*blockDim.x + job.start.x;
	unsigned int y = threadIdx.y + blockIdx.y*blockDim.y + job.start.y;
	unsigned int z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
        REAL A1, A2, A3;

        REAL u, v, w, cc, d;

        d = get_Field_LAP(sw.d, x, y, z);
        u = get_Field_LAP(sw.u, x, y, z);
        v = get_Field_LAP(sw.v, x, y, z);
        w = get_Field_LAP(sw.w, x, y, z);
        cc = get_Field_LAP(sw.cc, x, y, z);

        tmp0 = d*tmp0;

		A1 = get_Field_LAP(sw.Ax, x, y, z);
		A2 = get_Field_LAP(sw.Ay, x, y, z);
		A3 = get_Field_LAP(sw.Az, x, y, z); 

        Stager_Warming_ker(x, y, z, tmp0, u, v, w, cc, fp, fm, A1, A2, A3, split_C1, split_C3);
	}
}

void Stager_Warming_out(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, int flag, cudaStream_t *stream){
	dim3 blockdim , griddim, size;

	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x+2*LAP, size.y+2*LAP, size.z+2*LAP);
    cudaJobPackage job( dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP), 
	                    dim3(job_in.end.x + LAP, job_in.end.y + LAP, job_in.end.z + LAP) );
    if(flag == 1){
        sw_split_out sw = {*pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAkx_d, *pAky_d, *pAkz_d};
    }else if(flag == 2){
        sw_split_out sw = {*pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAix_d, *pAiy_d, *pAiz_d};
    }else if(flag == 3){
        sw_split_out sw = {*pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAsx_d, *pAsy_d, *pAsz_d};
    }
    	
	CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker_out<<<griddim , blockdim, 0, *stream>>>(sw, *fp, *fm, tmp0, split_C1, split_C3, job) ));
}*/


#ifdef __cplusplus
}
#endif

// Filtering, to remove high-wavenumber oscillations
// Bogey C, Bailly C,  J. Comput. Phys. 194 (2004) 194-214
#include "parameters.h"
#include "parameters_d.h"
#include "utility.h"
#include "stdio.h"
#include "OCFD_mpi.h"
#include "math.h"
#include "OCFD_filtering.h"
#include "OCFD_mpi_dev.h"
#include "cuda_utility.h"



#ifdef __cplusplus
extern "C"{
#endif


void set_para_filtering(){
    // ib , ie filed without LAP
    // input ib <= i < ie
    // ib start from 0
	if(my_id == 0) printf("filter parameters readed\n");
		
	int ib, ie, jb, je, kb, ke;
    int node_ib, node_ie, node_jb, node_je, node_kb, node_ke;
    for(int k=0;k<NFiltering;k++){
		if(Filter_para[k][1] == 1) fiter_judge_X = 1;
		if(Filter_para[k][2] == 1) fiter_judge_Y = 1;
		if(Filter_para[k][3] == 1) fiter_judge_Z = 1;

        int flag_i = Filter_para[k][1], flag_j = Filter_para[k][2], flag_k = Filter_para[k][3];

        get_i_node(Filter_para[k][4], &node_ib, &ib);
        get_i_node(Filter_para[k][5], &node_ie, &ie);
        get_j_node(Filter_para[k][6], &node_jb, &jb);
        get_j_node(Filter_para[k][7], &node_je, &je);
        get_k_node(Filter_para[k][8], &node_kb, &kb);
        get_k_node(Filter_para[k][9], &node_ke, &ke);

        if(node_ib < npx) ib = 0;
        if(node_ib > npx) flag_i = 0;
        if(node_ie > npx) ie = nx;
        if(node_ie < npx) flag_i = 0;
        if(npx == 0 && Iperiodic[0] != 1) ib=MAX(ib, 6);
        if(npx == NPX0-1 && Iperiodic[0] != 1) ie=MIN(ie, nx-5);

        if(node_jb < npy) jb = 0;
        if(node_jb > npy) flag_j = 0;
        if(node_je > npy) je = ny;
        if(node_je < npy) flag_j = 0;
        if(npy == 0 && Iperiodic[1] != 1) jb=MAX(jb, 6);
        if(npy == NPY0-1 && Iperiodic[1] != 1) je=MIN(je, ny-5);

        if(node_kb < npz) kb = 0;
        if(node_kb > npz) flag_k = 0;
        if(node_ke > npz) ke = nz;
        if(node_ke < npz) flag_k = 0;
        if(npz == 0 && Iperiodic[2] != 1) kb=MAX(kb, 6);
        if(npz == NPZ0-1 && Iperiodic[2] != 1) ke=MIN(ke, nz-5);
		
        Filter_para[k][1] = flag_i;
		Filter_para[k][2] = flag_j;
		Filter_para[k][3] = flag_k;
		Filter_para[k][4] = ib;
        Filter_para[k][5] = ie;
        Filter_para[k][6] = jb;
        Filter_para[k][7] = je;
        Filter_para[k][8] = kb;
        Filter_para[k][9] = ke;
    }
}

void filtering(
	REAL *pf,
	REAL *pf0,
	REAL *pp)
{
	int m, ib, ie, jb, je, kb, ke, IF_filter, Filter_scheme;
	REAL s0, rth;

	IF_filter = 0;

	for (m = 0; m < NFiltering; m++)
	{
		if (Istep % Filter_para[m][0] == 0)
			IF_filter = 1;
	}

	if (IF_filter == 0)
		return; // do not filtering in this step

	MPI_Barrier(MPI_COMM_WORLD);
	//  --------------Filtering --------------------

	if(fiter_judge_X == 1){
		exchange_boundary_x_packed_dev(pP , pP_d, Iperiodic[0]);

		for(int n=0;n<NVARS;n++){
			cudaField tmp;
			int size = pf_lap_d->pitch * ny_2lap * nz_2lap;
			tmp.pitch = pf_lap_d->pitch;
			tmp.ptr = pf_lap_d->ptr + n*size;
			exchange_boundary_x_packed_dev(pP, &tmp, Iperiodic[0]);
		}
	}


	if(fiter_judge_Y == 1){
		exchange_boundary_y_packed_dev(pP, pP_d, Iperiodic[1]);

		for(int n=0; n < NVARS; n++){
			cudaField tmp;
			int size = pf_lap_d->pitch * ny_2lap * nz_2lap;
			tmp.pitch = pf_lap_d->pitch;
			tmp.ptr = pf_lap_d->ptr + n*size;
			exchange_boundary_y_packed_dev(pP, &tmp, Iperiodic[1]);
		}
	}


	if(fiter_judge_Z == 1){
		exchange_boundary_z_packed_dev(pP, pP_d, Iperiodic[2]);

		for(int n=0; n < NVARS; n++){
			cudaField tmp;
			int size = pf_lap_d->pitch * ny_2lap * nz_2lap;
			tmp.pitch = pf_lap_d->pitch;
			tmp.ptr = pf_lap_d->ptr + n*size;
			exchange_boundary_z_packed_dev(pP, &tmp, Iperiodic[2]);
		}
	}


	for (m = 0; m < NFiltering; m++)
	{
		if(tt <= Filter_rpara[m][2]){
			if (Istep % Filter_para[m][0] == 0)
			{
                if (my_id == 0)
			    printf("filtering ......\n");
				ib = Filter_para[m][4];
				ie = Filter_para[m][5];
				jb = Filter_para[m][6];
				je = Filter_para[m][7];
				kb = Filter_para[m][8];
				ke = Filter_para[m][9];
				Filter_scheme = Filter_para[m][10];

				s0 = Filter_rpara[m][0];
				rth = Filter_rpara[m][1];
			
				if (Filter_scheme == Filter_Fo9p)
				{
					filter_x3d(pf, pf0, s0, ib, ie, jb, je, kb, ke);
				}
				else if (Filter_scheme == Filter_Fopt_shock)
				{
					filter_x3d_shock(pf_d, pf_lap_d, pP_d, s0, rth, ib, ie, jb, je, kb, ke, Filter_para[m][1]);
				}
			
				if (Filter_scheme == Filter_Fo9p)
				{
					filter_y3d(pf, pf0, s0, ib, ie, jb, je, kb, ke);
				}
				else if (Filter_scheme == Filter_Fopt_shock)
				{
					filter_y3d_shock(pf_d, pf_lap_d, pP_d, s0, rth, ib, ie, jb, je, kb, ke, Filter_para[m][2]);
				}
			
				if (Filter_scheme == Filter_Fo9p)
				{
					filter_z3d(pf, pf0, s0, ib, ie, jb, je, kb, ke);
				}
				else if (Filter_scheme == Filter_Fopt_shock)
				{
					filter_z3d_shock(pf_d, pf_lap_d, pP_d, s0, rth, ib, ie, jb, je, kb, ke, Filter_para[m][3]);
				}
			}
		}
	}
}

#define CUDA_FUN_UNFINISH \
if(my_id == 0){\
	printf("ERROR : %s ( File %s , Line %d ) , is undering developing , current unavailable!!!\n" , __FUNCTION__ , __FILE__,__LINE__);\
	MPI_Abort(MPI_COMM_WORLD,1);\
}
//---------------------------------------------------
void filter_x3d(
	REAL *pf,
	REAL *pf0,
	REAL s0,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke)
{
	CUDA_FUN_UNFINISH
/*
	int i, j, k, m, ib1 = ib, ie1 = ie;

	REAL(*f)
	[nz][ny][nx] = PTR2ARRAY3(pf, nx, ny, nz);
	REAL(*f0)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pf0, nx + 2 * LAP, ny + 2 * LAP);

	const REAL d0 = 0.243527493120, d1 = -0.204788880640, d2 = 0.120007591680;
	const REAL d3 = -0.045211119360, d4 = 0.008228661760;

	if (npx == 0 && Iperiodic[0] != 1)
		ib1 = MAX(ib, 6);
	if (npx == NPX0 - 1 && Iperiodic[0] != 1)
		ie1 = MIN(ie, nx - 5);

	for (m = 0; m < NVARS; m++)
	{IF_Filter_X
		for (k = 0; k < nz; k++)
		{
			for (j = 0; i < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					f0[k + LAP][j + LAP][i + LAP] = f[m][k][j][i];
				}
			}
		}

		exchange_boundary_x(pf0, Iperiodic[0]);

		for (k = kb + LAP; k <= ke + LAP; k++)
		{
			for (j = jb + LAP; j <= je + LAP; j++)
			{
				for (i = ib1 + LAP; i <= ie1 + LAP; i++)
				{
					f[m][k - LAP][j - LAP][i - LAP] = f0[k][j][i] - s0 * (d0 * f0[k][j][i] + d1 * (f0[k][j][i - 1] + f0[k][j][i + 1]) + d2 * (f0[k][j][i - 2] + f0[k][j][i + 2]) + d3 * (f0[k][j][i - 3] + f0[k][j][i + 3]) + d4 * (f0[k][j][i - 4] + f0[k][j][i + 4]));
				}
			}
		}
	}
*/
}

//---------------------------------------------------
void filter_y3d(
	REAL *pf,
	REAL *pf0,
	REAL s0,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke)
{
	CUDA_FUN_UNFINISH
/*
	int i, j, k, m, jb1 = jb, je1 = je;

	REAL(*f)
	[nz][ny][nx] = PTR2ARRAY3(pf, nx, ny, nz);
	REAL(*f0)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pf0, nx + 2 * LAP, ny + 2 * LAP);

	const REAL d0 = 0.243527493120, d1 = -0.204788880640, d2 = 0.120007591680;
	const REAL d3 = -0.045211119360, d4 = 0.008228661760;

	if (npy == 0 && Iperiodic[1] != 1)
		jb1 = MAX(jb, 6);
	if (npy == NPY0 - 1 && Iperiodic[1] != 1)
		je1 = MIN(je, ny - 5);

	for (m = 0; m < NVARS; m++)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; i < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					f0[k + LAP][j + LAP][i + LAP] = f[m][k][j][i];
				}
			}
		}

		exchange_boundary_y(pf0, Iperiodic[1]);

		for (k = kb + LAP; k <= ke + LAP; k++)
		{
			for (j = jb1 + LAP; j <= je1 + LAP; j++)
			{
				for (i = ib + LAP; i <= ie + LAP; i++)
				{
					f[m][k - LAP][j - LAP][i - LAP] = f0[k][j][i] - s0 * (d0 * f0[k][j][i] + d1 * (f0[k][j - 1][i] + f0[k][j + 1][i]) + d2 * (f0[k][j - 2][i] + f0[k][j + 2][i]) + d3 * (f0[k][j - 3][i] + f0[k][j + 3][i]) + d4 * (f0[k][j - 4][i] + f0[k][j + 4][i]));
				}
			}
		}
	}
*/
}

//---------------------------------------------------
void filter_z3d(
	REAL *pf,
	REAL *pf0,
	REAL s0,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke)
{
	CUDA_FUN_UNFINISH
/*
	int i, j, k, m, kb1 = kb, ke1 = ke;

	REAL(*f)
	[nz][ny][nx] = PTR2ARRAY3(pf, nx, ny, nz);
	REAL(*f0)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pf0, nx + 2 * LAP, ny + 2 * LAP);

	const REAL d0 = 0.243527493120, d1 = -0.204788880640, d2 = 0.120007591680;
	const REAL d3 = -0.045211119360, d4 = 0.008228661760;

	if (npz == 0 && Iperiodic[2] != 1)
		kb1 = MAX(kb, 6);
	if (npz == NPZ0 - 1 && Iperiodic[2] != 1)
		ke1 = MIN(ke, nz - 5);

	for (m = 0; m < NVARS; m++)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; i < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					f0[k + LAP][j + LAP][i + LAP] = f[m][k][j][i];
				}
			}
		}

		exchange_boundary_z(pf0, Iperiodic[2]);

		for (k = kb1 + LAP; k <= ke1 + LAP; k++)
		{
			for (j = jb + LAP; j <= je + LAP; j++)
			{
				for (i = ib + LAP; i <= ie + LAP; i++)
				{
					f[m][k - LAP][j - LAP][i - LAP] = f0[k][j][i] - s0 * (d0 * f0[k][j][i] + d1 * (f0[k - 1][k][i] + f0[k + 1][j][i]) + d2 * (f0[k - 2][k][i] + f0[k + 2][j][i]) + d3 * (f0[k - 3][k][i] + f0[k + 3][j][i]) + d4 * (f0[k - 4][k][i] + f0[k + 4][j][i]));
				}
			}
		}
	}
*/
}
//------------------------------------------------------------
// Shock cpaturing filtering

static __device__ __constant__ REAL filter_shock_c1_d = -0.2103830;
static __device__ __constant__ REAL filter_shock_c2_d = 0.0396170;

__global__ void filter_x3d_shock_kernel(cudaField P , cudaSoA f_lap , cudaSoA f , REAL rth , REAL s0 , cudaJobPackage job){
	// eyes on field WITH lap
	// blockdim.x = ie - ib + 2
	// job.size.x = filted size
    unsigned int x = (blockDim.x-2) * blockIdx.x + threadIdx.x - 1 + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	extern __shared__ REAL hh[];
    if( x<job.end.x + 1 && y<job.end.y && z<job.end.z){
		{
			REAL ri;
			{
				REAL dp0 , dp1 , dp2 , p_00;
				REAL p_m2 , p_m1 , p_p1 , p_p2;
				p_m2 = get_Field_LAP(P , x-2,y,z);
				p_m1 = get_Field_LAP(P , x-1,y,z);
				p_00 = get_Field_LAP(P , x  ,y,z);
				p_p1 = get_Field_LAP(P , x+1,y,z);
				p_p2 = get_Field_LAP(P , x+2,y,z);

				dp0 = 0.25 * (-p_p1 + 2.0 * p_00 - p_m1);
				dp1 = 0.25 * (-p_p2 + 2.0 * p_p1 - p_00);
				dp2 = 0.25 * (-p_00 + 2.0 * p_m1 - p_m2);
				ri = 0.5 * ((dp0 - dp1) * (dp0 - dp1) + (dp0 - dp2) * (dp0 - dp2)) / (p_00 * p_00) + 1e-16;
			}
			unsigned lid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
			ri = 1.0 - rth / ri;
			hh[lid] = 0.5 * (ri + fabs(ri));
		}
	}
	__syncthreads();
    if( threadIdx.x > 1 && x<job.end.x && y<job.end.y && z<job.end.z){
		REAL Sc1 , Sc2;
		{
			unsigned lid =threadIdx.x -1 + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
			Sc1 = 0.5*(hh[lid] + hh[lid+1]) * s0;
			Sc2 = 0.5*(hh[lid] + hh[lid-1]) * s0;
		}
		x -= 1;
		for(char m = 0;m<NVARS;m++){
			get_SoA(f , x-LAP , y-LAP , z-LAP , m) = 
				get_SoA_LAP(f_lap , x,y,z, m) - (
					Sc1 * ( filter_shock_c1_d * (get_SoA_LAP(f_lap , x+1,y,z, m) - get_SoA_LAP(f_lap, x,y,z, m)    ) + filter_shock_c2_d * (get_SoA_LAP(f_lap ,x+2 , y,z , m) - get_SoA_LAP(f_lap , x-1 , y,z, m)) )
				  - Sc2 * ( filter_shock_c1_d * (get_SoA_LAP(f_lap , x,y,z, m)   - get_SoA_LAP(f_lap, x-1,y,z , m) ) + filter_shock_c2_d * (get_SoA_LAP(f_lap ,x+1 , y,z , m) - get_SoA_LAP(f_lap , x-2 , y,z, m)) )
				);
		}
	}
}

void filter_x3d_shock(
	cudaSoA *pf,
	cudaSoA *pf0,
	cudaField *pp,
	REAL s0,
	REAL rth,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke,
	int IF_Filter)
{
	if(IF_Filter == 1) 
	{
		ib += LAP;
		ie += LAP;
		jb += LAP;
		je += LAP;
		kb += LAP;
		ke += LAP;

		dim3 griddim , blockdim;
		cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , ie - ib , je - jb , ke - kb);
		blockdim.x += 2;
		cudaJobPackage job(dim3(ib , jb , kb),dim3(ie , je , ke));

		CUDA_LAUNCH(( filter_x3d_shock_kernel<<<griddim , blockdim , sizeof(REAL)*(blockdim.x) * blockdim.y * blockdim.z>>>(*pP_d, *pf_lap_d, *pf_d, rth, s0, job) ));
	}
}

//---------------------------------------------------
__global__ void filter_y3d_shock_kernel(cudaField P, cudaSoA f_lap, cudaSoA f, REAL rth, REAL s0, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = (blockDim.y-2) * blockIdx.y + threadIdx.y - 1 + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	extern __shared__ REAL hh[];
	if( x<job.end.x && y<job.end.y + 1 && z<job.end.z){
		{
			REAL ri;
			{
				REAL dp0 , dp1 , dp2 , p_00;
				REAL p_m2 , p_m1 , p_p1 , p_p2;
				p_m2 = get_Field_LAP(P , x,y-2,z);
				p_m1 = get_Field_LAP(P , x,y-1,z);
				p_00 = get_Field_LAP(P , x  ,y,z);
				p_p1 = get_Field_LAP(P , x,y+1,z);
				p_p2 = get_Field_LAP(P , x,y+2,z);

				dp0 = 0.25 * (-p_p1 + 2.0 * p_00 - p_m1);
				dp1 = 0.25 * (-p_p2 + 2.0 * p_p1 - p_00);
				dp2 = 0.25 * (-p_00 + 2.0 * p_m1 - p_m2);
				ri = 0.5 * ((dp0 - dp1) * (dp0 - dp1) + (dp0 - dp2) * (dp0 - dp2)) / (p_00 * p_00) + 1e-16;
			}
			unsigned lid = threadIdx.y + blockDim.y * (threadIdx.x + blockDim.x * threadIdx.z);
			ri = 1.0 - rth / ri;
			hh[lid] = 0.5 * (ri + fabs(ri));
		}
	}
	__syncthreads();
    if( x<job.end.x && threadIdx.y > 1 && y<job.end.y && z<job.end.z){
		REAL Sc1 , Sc2;
		{
			unsigned lid = threadIdx.y -1 + blockDim.y * (threadIdx.x + blockDim.x * threadIdx.z);
			Sc1 = 0.5*(hh[lid] + hh[lid+1]) * s0;
			Sc2 = 0.5*(hh[lid] + hh[lid-1]) * s0;
		}
		y -= 1;
		for(char m = 0;m<NVARS;m++){
			get_SoA(f , x-LAP , y-LAP , z-LAP , m) = 
				get_SoA_LAP(f_lap , x,y,z, m) - (
					Sc1 * ( filter_shock_c1_d * (get_SoA_LAP(f_lap , x,y+1,z, m) - get_SoA_LAP(f_lap, x,y,z, m)    ) + filter_shock_c2_d * (get_SoA_LAP(f_lap ,x , y+2,z , m) - get_SoA_LAP(f_lap , x , y-1,z, m)) )
				  - Sc2 * ( filter_shock_c1_d * (get_SoA_LAP(f_lap , x,y,z, m)   - get_SoA_LAP(f_lap, x,y-1,z , m) ) + filter_shock_c2_d * (get_SoA_LAP(f_lap ,x , y+1,z , m) - get_SoA_LAP(f_lap , x , y-2,z, m)) )
				);
		}
	}
}


void filter_y3d_shock(
	cudaSoA *pf,
	cudaSoA *pf0,
	cudaField *pp,
	REAL s0,
	REAL rth,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke,
	int IF_Filter)
{
	if(IF_Filter == 1){
		ib += LAP;
		ie += LAP;
		jb += LAP;
		je += LAP;
		kb += LAP;
		ke += LAP;

		dim3 griddim, blockdim;
		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, ie - ib, je - jb, ke - kb);
		blockdim.y += 2;
		cudaJobPackage job(dim3(ib, jb, kb),dim3(ie, je, ke));

		CUDA_LAUNCH((filter_y3d_shock_kernel<<<griddim, blockdim, sizeof(REAL)*(blockdim.x)*blockdim.y*blockdim.z>>>(*pP_d, *pf_lap_d, *pf_d, rth, s0, job)));
	}
}

//---------------------------------------------------
__global__ void filter_z3d_shock_kernel(cudaField P, cudaSoA f_lap, cudaSoA f, REAL rth, REAL s0, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = (blockDim.z-2) * blockIdx.z + threadIdx.z - 1 + job.start.z;
	extern __shared__ REAL hh[];
	if( x<job.end.x && y<job.end.y && z<job.end.z + 1){
		{
			REAL ri;
			{
				REAL dp0 , dp1 , dp2 , p_00;
				REAL p_m2 , p_m1 , p_p1 , p_p2;
				p_m2 = get_Field_LAP(P , x,y,z-2);
				p_m1 = get_Field_LAP(P , x,y,z-1);
				p_00 = get_Field_LAP(P , x  ,y,z);
				p_p1 = get_Field_LAP(P , x,y,z+1);
				p_p2 = get_Field_LAP(P , x,y,z+2);

				dp0 = 0.25 * (-p_p1 + 2.0 * p_00 - p_m1);
				dp1 = 0.25 * (-p_p2 + 2.0 * p_p1 - p_00);
				dp2 = 0.25 * (-p_00 + 2.0 * p_m1 - p_m2);
				ri = 0.5 * ((dp0 - dp1) * (dp0 - dp1) + (dp0 - dp2) * (dp0 - dp2)) / (p_00 * p_00) + 1e-16;
			}
			unsigned lid = threadIdx.z + blockDim.z * (threadIdx.x + blockDim.x * threadIdx.y);
			ri = 1.0 - rth / ri;
			hh[lid] = 0.5 * (ri + fabs(ri));
		}
	}
	__syncthreads();
    if( x<job.end.x && y<job.end.y && threadIdx.z > 1 && z<job.end.z){
		REAL Sc1 , Sc2;
		{
			unsigned lid = threadIdx.z -1 + blockDim.z * (threadIdx.x + blockDim.x * threadIdx.y);
			Sc1 = 0.5*(hh[lid] + hh[lid+1]) * s0;
			Sc2 = 0.5*(hh[lid] + hh[lid-1]) * s0;
		}
		z -= 1;
		for(char m = 0;m<NVARS;m++){
			get_SoA(f , x-LAP , y-LAP , z-LAP , m) = 
				get_SoA_LAP(f_lap , x,y,z, m) - (
					Sc1 * ( filter_shock_c1_d * (get_SoA_LAP(f_lap , x,y,z+1, m) - get_SoA_LAP(f_lap, x,y,z, m)    ) + filter_shock_c2_d * (get_SoA_LAP(f_lap ,x , y,z+2 , m) - get_SoA_LAP(f_lap , x , y,z-1, m)) )
				  - Sc2 * ( filter_shock_c1_d * (get_SoA_LAP(f_lap , x,y,z, m)   - get_SoA_LAP(f_lap, x,y,z-1 , m) ) + filter_shock_c2_d * (get_SoA_LAP(f_lap ,x , y,z+1 , m) - get_SoA_LAP(f_lap , x , y,z-2, m)) )
				);
		}
	}
}


void filter_z3d_shock(
	cudaSoA *pf,
	cudaSoA *pf0,
	cudaField *pp,
	REAL s0,
	REAL rth,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke,
	int IF_Filter)
{
	if(IF_Filter == 1){
		ib += LAP;
		ie += LAP;
		jb += LAP;
		je += LAP;
		kb += LAP;
		ke += LAP;

		dim3 griddim, blockdim;
		cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, ie - ib, je - jb, ke - kb);
		blockdim.z += 2;
		cudaJobPackage job(dim3(ib, jb, kb),dim3(ie, je, ke));

		CUDA_LAUNCH((filter_y3d_shock_kernel<<<griddim, blockdim, sizeof(REAL)*(blockdim.x)*blockdim.y*blockdim.z>>>(*pP_d, *pf_lap_d, *pf_d, rth, s0, job)));
	}
}
//------------------------------------------------------------


#ifdef __cplusplus
}
#endif

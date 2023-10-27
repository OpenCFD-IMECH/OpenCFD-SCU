/*OpenCFD ver 1.4, CopyRight by Li Xinliang, LNM, Institute of Mechanics, CAS, Beijing, Email: lixl@lnm.imech.ac.cn
MPI Subroutines, such as computational domain partation, MPI message send and recv   
只支持N_MSG_SIZE=0, -2  两种通信方式 
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "utility.h"
#include "OCFD_mpi.h"
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif

static REAL *BUFFER_MPI;// Buffer for MPI  message transfer (used by MPI_Bsend)
static int IBUFFER_SIZE = 100000;

static char mpi_mem_initialized = 0;
void opencfd_mem_init_mpi(){
    // mpi corrosponding data
	if(mpi_mem_initialized == 0){
		mpi_mem_initialized = 1;
		i_nn = (int *)malloc(sizeof(int) * NPX0);
		i_offset = (int *)malloc(sizeof(int) * NPX0);

		j_nn = (int *)malloc(sizeof(int) * NPY0);
		j_offset = (int *)malloc(sizeof(int) * NPY0);

		k_nn = (int *)malloc(sizeof(int) * NPZ0);
		k_offset = (int *)malloc(sizeof(int) * NPZ0);
	}
}

void opencfd_mem_finalize_mpi(){
	// 仅仅用于free内存
	if(mpi_mem_initialized == 1){
		mpi_mem_initialized = 0;
		free(i_nn);
		free(j_nn);
		free(k_nn);

		free(i_offset);
		free(j_offset);
		free(k_offset);

		// MPI_Type_free(&TYPE_LAPX2);
		// MPI_Type_free(&TYPE_LAPZ2);
		// MPI_Type_free(&TYPE_LAPY2);

		MPI_Comm_free(&MPI_COMM_X);
		MPI_Comm_free(&MPI_COMM_Y);
		MPI_Comm_free(&MPI_COMM_Z);
		MPI_Comm_free(&MPI_COMM_XY);
		MPI_Comm_free(&MPI_COMM_XZ);
		MPI_Comm_free(&MPI_COMM_YZ);
	}
}

void mpi_init(int *Argc , char *** Argv){
	int flag, provided;
	MPI_Initialized(&flag);
	if(!flag){
		MPI_Init_thread(Argc, Argv, MPI_THREAD_MULTIPLE, &provided);
		if(provided != MPI_THREAD_MULTIPLE){
                    printf("\033[31mMPI do not Support Multiple thread\033[0m\n");
                    exit(0);
		}
		MPI_Comm_rank(MPI_COMM_WORLD , &my_id);

    	thread_handles = (pthread_t* )malloc(12*sizeof(pthread_t));
		BUFFER_MPI = (REAL*)malloc(sizeof(REAL)*IBUFFER_SIZE);
		MPI_Buffer_attach(BUFFER_MPI , IBUFFER_SIZE*sizeof(REAL));
	}
}
void mpi_finalize(){
	int flag;
	MPI_Initialized(&flag);
	if(flag){
		opencfd_mem_finalize_mpi();

		MPI_Buffer_detach(BUFFER_MPI , &IBUFFER_SIZE);
		free(BUFFER_MPI);
		MPI_Finalize();

		free(thread_handles);
	}
}

void part()
{	
	// Domain partation----------------------------------------------------------------------------
	int k, ka;
	int npx1, npy1, npz1, npx2, npy2, npz2;

	// ---------------------------------------------------------------------------------------------
	int np_size;
	MPI_Comm_size(MPI_COMM_WORLD, &np_size);
	if (np_size != NPX0 * NPY0 * NPZ0)
	{
		if (my_id == 0)
			printf("The Number of total Processes is not equal to NPX0*NPY0*NPZ0 !\n");
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	// 直接进行手动分块,确定笛卡尔网格对应的进程号.将rank按照x,y,z的维度进行分解
	npx = my_id % NPX0;  //x方向进程ID
	npy = my_id % (NPX0 * NPY0) / NPX0;  //y方向进程ID
	npz = my_id / (NPX0 * NPY0);  //z方向进程ID
	// ------commonicators-----------------------------------------------------------------------------

	MPI_Comm_split(MPI_COMM_WORLD, npz * NPX0 * NPY0 + npy * NPX0, npx, &MPI_COMM_X); // 1-D
	MPI_Comm_split(MPI_COMM_WORLD, npz * NPX0 * NPY0 + npx, npy, &MPI_COMM_Y);
	MPI_Comm_split(MPI_COMM_WORLD, npy * NPX0 + npx, npz, &MPI_COMM_Z);
	MPI_Comm_split(MPI_COMM_WORLD, npz, npy * NPX0 + npx, &MPI_COMM_XY); // 2-D
	MPI_Comm_split(MPI_COMM_WORLD, npy, npz * NPX0 + npx, &MPI_COMM_XZ);
	MPI_Comm_split(MPI_COMM_WORLD, npx, npz * NPY0 + npy, &MPI_COMM_YZ);

	// ------------------------------------------------------------------------------------------------
	// 均匀分配网格， 如果NX_GLOBAL不能被NPX0整除，将余下的网格点分到靠前的节点
	// ------------------------------------------------------------------------------------------------
	nx = NX_GLOBAL / NPX0;
	ny = NY_GLOBAL / NPY0;
	nz = NZ_GLOBAL / NPZ0;
	if (npx < NX_GLOBAL % NPX0)
		nx = nx + 1;
	if (npy < NY_GLOBAL % NPY0)
		ny = ny + 1;
	if (npz < NZ_GLOBAL % NPZ0)
		nz = nz + 1;


	// ------npx=k的节点上x方向网格点的个数，起始位置
	// --------------------------------------------------------------------
	for (k = 0; k < NPX0; k++)
	{
		ka = fmin(k, NX_GLOBAL % NPX0);
		// offset为当前处理器所计算的最大下标
		// offset实际提供了全局的分块信息
		i_offset[k] = NX_GLOBAL / NPX0 * k + ka;
		// nn提供全局的分块大小信息
		i_nn[k] = NX_GLOBAL / NPX0;
		if (k < NX_GLOBAL % NPX0)
			i_nn[k] += 1;
	}
	for (k = 0; k < NPY0; k++)
	{
		ka = fmin(k, NY_GLOBAL % NPY0);
		j_offset[k] = NY_GLOBAL / NPY0 * k + ka;
		j_nn[k] = NY_GLOBAL / NPY0;
		if (k < NY_GLOBAL % NPY0)
			j_nn[k] += 1;
	}
	for (k = 0; k < NPZ0; k++)
	{
		ka = fmin(k, NZ_GLOBAL % NPZ0);
		k_offset[k] = NZ_GLOBAL / NPZ0 * k + ka;
		k_nn[k] = NZ_GLOBAL / NPZ0;
		if (k < NZ_GLOBAL % NPZ0)
			k_nn[k] += 1;
	}
	// --------------------------------------------------------------------------------
	// -------New Data TYPE------------------------------------------------------------
	New_MPI_datatype();

	// --------define proc id:  the right, left, up, bottom, front and backward  procs
	npx1 = my_mod1(npx - 1, NPX0);
	npx2 = my_mod1(npx + 1, NPX0);
	// 利用comm_world的全局下标查找邻居
	ID_XM1 = npz * (NPX0 * NPY0) + npy * NPX0 + npx1; // -1 proc in x-direction
	ID_XP1 = npz * (NPX0 * NPY0) + npy * NPX0 + npx2; // +1 proc in x-direction
	if (Iperiodic[0] == 0 && npx == 0)
		ID_XM1 = MPI_PROC_NULL; // if not periodic, 0 node donot send mesg to NPX0-1 node
	if (Iperiodic[0] == 0 && npx == NPX0 - 1)
		ID_XP1 = MPI_PROC_NULL;

	npy1 = my_mod1(npy - 1, NPY0);
	npy2 = my_mod1(npy + 1, NPY0);
	ID_YM1 = npz * (NPX0 * NPY0) + npy1 * NPX0 + npx;
	ID_YP1 = npz * (NPX0 * NPY0) + npy2 * NPX0 + npx;
	if (Iperiodic[1] == 0 && npy == 0)
		ID_YM1 = MPI_PROC_NULL; // if not periodic, 0 node donot send mesg to NPY0-1 node
	if (Iperiodic[1] == 0 && npy == NPY0 - 1)
		ID_YP1 = MPI_PROC_NULL;

	npz1 = my_mod1(npz - 1, NPZ0);
	npz2 = my_mod1(npz + 1, NPZ0);
	ID_ZM1 = npz1 * (NPX0 * NPY0) + npy * NPX0 + npx;
	ID_ZP1 = npz2 * (NPX0 * NPY0) + npy * NPX0 + npx;
	if (Iperiodic[2] == 0 && npz == 0)
		ID_ZM1 = MPI_PROC_NULL; // if not periodic, 0 node donot send mesg to NPZ0-1 node
	if (Iperiodic[2] == 0 && npz == NPZ0 - 1)
		ID_ZP1 = MPI_PROC_NULL;

	// --------------------------------------------------------------
	MPI_Barrier(MPI_COMM_WORLD);
}

// --------------------------------------------------------------------------------
int my_mod1(int i, int n)
{
	if (i < 0)
	{
		return i + n;
	}
	else if (i > n - 1)
	{
		return i - n;
	}
	else
	{
		return i;
	}
}
// -----------------------------------------------------------------------------------------------
//  Send Recv non-continuous data using derivative data type
void New_MPI_datatype()
{
	MPI_Type_vector(ny, LAP, nx + 2 * LAP, OCFD_DATA_TYPE, &TYPE_LAPX1); //[0:LAP,LAP:ny+LAP,k]
	MPI_Type_commit(&TYPE_LAPX1);
	MPI_Type_create_hvector(nz, 1, (nx + 2 * LAP) * (ny + 2 * LAP) * sizeof(REAL), TYPE_LAPX1, &TYPE_LAPX2); //[0:LAP,LAP:ny+LAP,LAP:nz+LAP]

	MPI_Type_vector(LAP, nx, nx + 2 * LAP, OCFD_DATA_TYPE, &TYPE_LAPY1); //[LAP:nx+LAP,0:LAP,K]
	MPI_Type_commit(&TYPE_LAPY1);
	MPI_Type_create_hvector(nz, 1, (nx + 2 * LAP) * (ny + 2 * LAP) * sizeof(REAL), TYPE_LAPY1, &TYPE_LAPY2); //[LAP:nx+LAP,0:LAP,LAP:nz+LAP]

	MPI_Type_vector(ny, nx, nx + 2 * LAP, OCFD_DATA_TYPE, &TYPE_LAPZ1);
	MPI_Type_commit(&TYPE_LAPZ1);
	MPI_Type_create_hvector(LAP, 1, (nx + 2 * LAP) * (ny + 2 * LAP) * sizeof(REAL), TYPE_LAPZ1, &TYPE_LAPZ2);

	MPI_Type_commit(&TYPE_LAPX2);
	MPI_Type_commit(&TYPE_LAPY2);
	MPI_Type_commit(&TYPE_LAPZ2);

	MPI_Type_free(&TYPE_LAPX1);
	MPI_Type_free(&TYPE_LAPY1);
	MPI_Type_free(&TYPE_LAPZ1);

	MPI_Barrier(MPI_COMM_WORLD);
}
// -----------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------
// ----Form a Global index, get the node information and local index
void get_i_node(int i_global, int *node_i, int *i_local)   //输入全局的坐标，定位它在哪一个节点，并定位它在此节点的当地坐标
{
	int ia;
	*node_i = NPX0 - 1;
	for (ia = 0; ia < NPX0 - 1; ia++)
	{
		if (i_global >= i_offset[ia] && i_global < i_offset[ia + 1])
			*node_i = ia;
	}
	*i_local = i_global - i_offset[*node_i];
}
// -------------------------------------------------------------------------------
void get_j_node(int j_global, int *node_j, int *j_local)
{
	int ja;
	*node_j = NPY0 - 1;
	for (ja = 0; ja < NPY0 - 1; ja++)
	{
		if (j_global >= j_offset[ja] && j_global < j_offset[ja + 1])
			*node_j = ja;
	}
	*j_local = j_global - j_offset[*node_j];
}
// -----------------------------------------------------------------------------------
void get_k_node(int k_global, int *node_k, int *k_local)
{
	int ka;
	*node_k = NPZ0 - 1;
	for (ka = 0; ka < NPZ0 - 1; ka++)
	{
		if (k_global >= k_offset[ka] && k_global < k_offset[ka + 1])//<=  >  <
			*node_k = ka;
	}
	*k_local = k_global - k_offset[*node_k];
}

// !------------------------------------------------------------------------------------
int get_id(int npx1, int npy1, int npz1)
{
	return npz1 * (NPX0 * NPY0) + npy1 * NPX0 + npx1;
}
// -------------------------------------------------------------------------------------
// Message send and recv at inner boundary (or 'MPI boundary')
void exchange_boundary_xyz(REAL *pf)
{
	exchange_boundary_x(pf, Iperiodic[0]);
	exchange_boundary_y(pf, Iperiodic[1]);
	exchange_boundary_z(pf, Iperiodic[2]);
}
// ----------------------------------------------------------------------------------------
void exchange_boundary_x(REAL *pf, int Iperiodic1)
{
	if (MSG_BLOCK_SIZE == 0)
	{
		exchange_boundary_x_standard(pf, Iperiodic1);
	}
	else if (MSG_BLOCK_SIZE == -2)
	{
		exchange_boundary_x_deftype(pf);
	}
	else
	{
		printf("MSG_BLOCK_SIZE error in exchange_boundary_x !");
	}
}
// -----------------------------------------------------------------------------------------------
void exchange_boundary_y(REAL *pf, int Iperiodic1)
{
	if (MSG_BLOCK_SIZE == 0)
	{
		exchange_boundary_y_standard(pf, Iperiodic1);
	}
	else if (MSG_BLOCK_SIZE == -2)
	{
		exchange_boundary_y_deftype(pf);
	}
	else
	{
		printf("MSG_BLOCK_SIZE error in exchange_boundary_y !");
	}
}
// -----------------------------------------------------------------------------------------------
void exchange_boundary_z(REAL *pf, int Iperiodic1)
{
	if (MSG_BLOCK_SIZE == 0)
	{
		exchange_boundary_z_standard(pf, Iperiodic1);
	}
	else if (MSG_BLOCK_SIZE == -2)
	{
		exchange_boundary_z_deftype(pf);
	}
	else
	{
		printf("MSG_BLOCK_SIZE error in exchange_boundary_z !");
	}
}
// =========================================================================================================
// Boundary message communication (exchange message)
// =========================================================================================================
// Standard (most used)
void exchange_boundary_x_standard(REAL *pf, int Iperiodic1)
{
	// send and recv mesg, to exchange_boundary array in x direction.
	// To avoid msg block, cutting long msg to short msgs
	MPI_Status status;
	int i, j, k, k1, nsize = LAP * ny * nz;
	// 1为左侧数据，2为右侧数据
	REAL tmp_send1[nsize], tmp_send2[nsize], tmp_recv1[nsize], tmp_recv2[nsize];
	REAL(*f)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pf, nx + 2 * LAP, ny + 2 * LAP);

	if (npx != 0 || Iperiodic1 == 1){
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < LAP; i++)
				{
					k1 = k * ny * LAP + j * LAP + i;
					tmp_send1[k1] = f[k + LAP][j + LAP][i + LAP];
				}
			}
		}
	}
	if (npx != NPX0 - 1 || Iperiodic1 == 1){
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < LAP; i++)
				{
					k1 = k * ny * LAP + j * LAP + i;
					tmp_send2[k1] = f[k + LAP][j + LAP][i + nx];
				}
			}
		}
	}
	MPI_Sendrecv(tmp_send1, nsize, OCFD_DATA_TYPE, ID_XM1, 9000,
				 tmp_recv2, nsize, OCFD_DATA_TYPE, ID_XP1, 9000,
				 MPI_COMM_WORLD, &status);
	MPI_Sendrecv(tmp_send2, nsize, OCFD_DATA_TYPE, ID_XP1, 8000,
				 tmp_recv1, nsize, OCFD_DATA_TYPE, ID_XM1, 8000,
				 MPI_COMM_WORLD, &status);

	//  if not periodic, node npx=0 Do Not need f(i-LAP,j,k)
	if (npx != 0 || Iperiodic1 == 1)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < LAP; i++)
				{
					k1 = k * ny * LAP + j * LAP + i;
					f[k + LAP][j + LAP][i] = tmp_recv1[k1];
				}
			}
		}
	}
	if (npx != NPX0 - 1 || Iperiodic1 == 1)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < LAP; i++)
				{
					k1 = k * ny * LAP + j * LAP + i;
					f[k + LAP][j + LAP][i + nx + LAP] = tmp_recv2[k1];
				}
			}
		}
	}
}
// ------------------------------------------------------
void exchange_boundary_y_standard(REAL *pf, int Iperiodic1)
{
	MPI_Status status;
	REAL(*f)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pf, nx + 2 * LAP, ny + 2 * LAP);
	int i, j, k, k1, nsize = LAP * nz* nx;
	// 1为下方，2为上方
	REAL tmp_send1[nsize], tmp_send2[nsize], tmp_recv1[nsize], tmp_recv2[nsize];

	if (npy != 0 || Iperiodic1 == 1)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < LAP; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * LAP * nx + j * nx + i;
					tmp_send1[k1] = f[k + LAP][j + LAP][i + LAP];
				}
			}
		}
	}
	if (npy != NPY0 - 1 || Iperiodic1 == 1)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < LAP; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * LAP * nx + j * nx + i;
					tmp_send2[k1] = f[k + LAP][j + ny][i + LAP];
				}
			}
		}
	}
	MPI_Sendrecv(tmp_send1, nsize, OCFD_DATA_TYPE, ID_YM1, 9000,
				 tmp_recv2, nsize, OCFD_DATA_TYPE, ID_YP1, 9000,
				 MPI_COMM_WORLD, &status);
	MPI_Sendrecv(tmp_send2, nsize, OCFD_DATA_TYPE, ID_YP1, 8000,
				 tmp_recv1, nsize, OCFD_DATA_TYPE, ID_YM1, 8000,
				 MPI_COMM_WORLD, &status);

	if (npy != 0 || Iperiodic1 == 1)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < LAP; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * LAP * nx + j * nx + i;
					f[k + LAP][j][i + LAP] = tmp_recv1[k1];
				}
			}
		}
	}
	if (npy != NPY0 - 1 || Iperiodic1 == 1)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < LAP; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * LAP * nx + j * nx + i;
					f[k + LAP][j + ny + LAP][i + LAP] = tmp_recv2[k1];
				}
			}
		}
	}
}
// ------------------------------------------------------------
void exchange_boundary_z_standard(REAL *pf, int Iperiodic1)
{
	MPI_Status status;
	REAL(*f)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pf, nx + 2 * LAP, ny + 2 * LAP);
	int i, j, k, k1, nsize = LAP * nx * ny;
	// 1为outward , 2为inward
	REAL tmp_send1[nsize], tmp_send2[nsize], tmp_recv1[nsize], tmp_recv2[nsize];

	if (npz != 0 || Iperiodic1 == 1){
		for (k = 0; k < LAP; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * nx * ny + j * nx + i;
					tmp_send1[k1] = f[k + LAP][j + LAP][i + LAP];
				}
			}
		}
	}
	if (npz != NPZ0 - 1 || Iperiodic1 == 1){
		for (k = 0; k < LAP; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * nx * ny + j * nx + i;
					tmp_send2[k1] = f[k + nz][j + LAP][i + LAP];
				}
			}
		}
	}
	MPI_Sendrecv(tmp_send1, nsize, OCFD_DATA_TYPE, ID_ZM1, 9000,
				 tmp_recv2, nsize, OCFD_DATA_TYPE, ID_ZP1, 9000,
				 MPI_COMM_WORLD, &status);
	MPI_Sendrecv(tmp_send2, nsize, OCFD_DATA_TYPE, ID_ZP1, 8000,
				 tmp_recv1, nsize, OCFD_DATA_TYPE, ID_ZM1, 8000,
				 MPI_COMM_WORLD, &status);

	if (npz != 0 || Iperiodic1 == 1)
	{
		for (k = 0; k < LAP; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * nx * ny + j * nx + i;
					f[k][j + LAP][i + LAP] = tmp_recv1[k1];
				}
			}
		}
	}

	if (npz != NPZ0 - 1 || Iperiodic1 == 1)
	{
		for (k = 0; k < LAP; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					k1 = k * nx * ny + j * nx + i;
					f[k + nz + LAP][j + LAP][i + LAP] = tmp_recv2[k1];
				}
			}
		}
	}
}
// ================================================================================
// -----------------------------------------------------------------------

//  mpi message send and recv, using user defined data type
void exchange_boundary_x_deftype(REAL *pf)
{
	MPI_Status status;
	MPI_Sendrecv(pf + idx2int(LAP, LAP, LAP), 1, TYPE_LAPX2, ID_XM1, 9000, pf + idx2int(nx + LAP, LAP, LAP), 1, TYPE_LAPX2, ID_XP1, 9000, MPI_COMM_WORLD, &status);
	MPI_Sendrecv(pf + idx2int(nx, LAP, LAP), 1, TYPE_LAPX2, ID_XP1, 8000, pf + idx2int(0, LAP, LAP), 1, TYPE_LAPX2, ID_XM1, 8000, MPI_COMM_WORLD, &status);
}
// ------------------------------------------------------
void exchange_boundary_y_deftype(REAL *pf)
{
	MPI_Status status;
	MPI_Sendrecv(pf + idx2int(LAP, LAP, LAP), 1, TYPE_LAPY2, ID_YM1, 9000, pf + idx2int(LAP, ny + LAP, LAP), 1, TYPE_LAPY2, ID_YP1, 9000, MPI_COMM_WORLD, &status);
	MPI_Sendrecv(pf + idx2int(LAP, ny, LAP), 1, TYPE_LAPY2, ID_YP1, 8000, pf + idx2int(LAP, 0, LAP), 1, TYPE_LAPY2, ID_YM1, 8000, MPI_COMM_WORLD, &status);
}
// ------------------------------------------------------------
void exchange_boundary_z_deftype(REAL *pf)
{
	MPI_Status status;
	MPI_Sendrecv(pf + idx2int(LAP, LAP, LAP), 1, TYPE_LAPZ2, ID_ZM1, 9000, pf + idx2int(LAP, LAP, nz + LAP), 1, TYPE_LAPZ2, ID_ZP1, 9000, MPI_COMM_WORLD, &status);
	MPI_Sendrecv(pf + idx2int(LAP, LAP, nx), 1, TYPE_LAPZ2, ID_ZP1, 8000, pf + idx2int(LAP, LAP, 0), 1, TYPE_LAPZ2, ID_ZM1, 8000, MPI_COMM_WORLD, &status);
}
// -----------------------------------------------------------------------


#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#include "parameters.h"
#include "utility.h"
#include "OCFD_mpi.h"
#include "io_warp.h"

//---------------------------------------------------
#ifdef __cplusplus
extern "C"{
#endif
//void write_2d_XYa(
//	FILE *file,
//	int ka,
//	int size_x,
//	int size_y,
//	int lap,
//	int *pU)
//{
//
//	int(*U)
//	[size_y + 2*lap][size_x + 2*lap] = (int(*)[size_y + 2*lap][size_x + 2*lap])(pU);
//	int(*U2d)
//	[NX_GLOBAL], (*U0)[NX_GLOBAL];
//	int node_k, k_local;
//
//	U2d = (int(*)[NX_GLOBAL])malloc(sizeof(int) * NX_GLOBAL * NY_GLOBAL);
//	memset((void*)U2d, 0, NX_GLOBAL * NY_GLOBAL * sizeof(int));
//	if (my_id == 0){
//		U0 = (int(*)[NX_GLOBAL])malloc(sizeof(int) * NX_GLOBAL * NY_GLOBAL);
//	}
//	//--------------------------------
//	get_k_node(ka, &node_k, &k_local);
//	k_local += lap;
//	int i, j;
//	if(npz == node_k){
//		for (j = lap; j < ny + lap; j++)
//		{
//			for (i = lap; i < nx + lap; i++)
//			{
//				U2d[j - lap + j_offset[npy]][i - lap + i_offset[npx]] = U[k_local][j][i];
//			}
//		}
//	}
//	MPI_Reduce(&U2d[0][0], &U0[0][0], NX_GLOBAL * NY_GLOBAL, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
////	if (my_id == 0)
////		FWRITE(U0, sizeof(REAL), NY_GLOBAL * NZ_GLOBAL, file)
//
//	if(my_id == 0){
//		for(j = 0; j < NY_GLOBAL; j++){
//			for(i = 0; i < NX_GLOBAL; i++){
//				fprintf(file, "%08d\n", U0[j][i]);
//			}
//		}
//	}
//
//	free(U2d);
//	if (my_id == 0)
//		free(U0);
//}
void write_2d_XY(
	FILE *file,
	int ka,
	int size_x,
	int size_y,
	int lap,
	int *pU,
	REAL *pU1)
{

	int(*U)[size_y + 2*lap][size_x + 2*lap] = (int(*)[size_y + 2*lap][size_x + 2*lap])(pU);
	REAL(*U1)[ny + 2*LAP][nx + 2*LAP] = (REAL(*)[ny + 2*LAP][nx + 2*LAP])(pU1);
	int(*U2d)[NX_GLOBAL], (*U0)[NX_GLOBAL];
	REAL(*U2d1)[NX_GLOBAL], (*U01)[NX_GLOBAL];
	int node_k, k_local;

	U2d = (int(*)[NX_GLOBAL])malloc(sizeof(int) * NX_GLOBAL * NY_GLOBAL);
	U2d1 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);
	memset((void*)U2d, 0, NX_GLOBAL * NY_GLOBAL * sizeof(int));
	memset((void*)U2d1, 0, NX_GLOBAL * NY_GLOBAL * sizeof(REAL));
	if (my_id == 0){
		U0 = (int(*)[NX_GLOBAL])malloc(sizeof(int) * NX_GLOBAL * NY_GLOBAL);
		U01 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);
	}
	//--------------------------------
	get_k_node(ka, &node_k, &k_local);
	k_local += lap;
	int i, j;
	if(npz == node_k){
		for (j = lap; j < ny + lap; j++)
		{
			for (i = lap; i < nx + lap; i++)
			{
				U2d[j - lap + j_offset[npy]][i - lap + i_offset[npx]] = U[k_local][j][i];
			}
		}

		for (j = LAP; j < ny + LAP; j++)
		{
			for (i = LAP; i < nx + LAP; i++)
			{
				U2d1[j - LAP + j_offset[npy]][i - LAP + i_offset[npx]] = U1[k_local + LAP][j][i];
			}
		}
	}
	MPI_Reduce(&U2d[0][0], &U0[0][0], NX_GLOBAL * NY_GLOBAL, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&U2d1[0][0], &U01[0][0], NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(my_id == 0){
		for(j = 0; j < NY_GLOBAL; j++){
			for(i = 0; i < NX_GLOBAL; i++){
				fprintf(file, "%08d%15.6lf\n", U0[j][i], U01[j][i]);
			}
		}
	}

	free(U2d);
	free(U2d1);
	if (my_id == 0){
		free(U0);
		free(U01);
	}
}


void write_2d_XYa(
	FILE *file,
	int ka,
	REAL *pU1)
{

	REAL(*U1)[ny + 2*LAP][nx + 2*LAP] = (REAL(*)[ny + 2*LAP][nx + 2*LAP])(pU1);
	REAL(*U2d1)[NX_GLOBAL], (*U01)[NX_GLOBAL];
	int node_k, k_local;

	U2d1 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);

	memset((void*)U2d1, 0, NX_GLOBAL * NY_GLOBAL * sizeof(REAL));

	if (my_id == 0){
		U01 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);
	}
	//--------------------------------
	get_k_node(ka, &node_k, &k_local);

	int i, j;
	if(npz == node_k){
		for (j = LAP; j < ny + LAP; j++)
		{
			for (i = LAP; i < nx + LAP; i++)
			{
				U2d1[j - LAP + j_offset[npy]][i - LAP + i_offset[npx]] = U1[k_local + LAP][j][i];
			}
		}
	}

	MPI_Reduce(&U2d1[0][0], &U01[0][0], NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
//	if (my_id == 0)
//		FWRITE(U0, sizeof(REAL), NY_GLOBAL * NZ_GLOBAL, file)

	if(my_id == 0) FWRITE(U01, sizeof(REAL), NX_GLOBAL * NY_GLOBAL, file)

	free(U2d1);
	if (my_id == 0){
		free(U01);
	}
}

void write_2d_YZa(
	FILE *file,
	int ia,
	REAL *pU1)
{

	REAL(*U1)[ny + 2*LAP][nx + 2*LAP] = (REAL(*)[ny + 2*LAP][nx + 2*LAP])(pU1);
	REAL(*U2d1)[NY_GLOBAL], (*U01)[NY_GLOBAL];
	int node_i, i_local;

	U2d1 = (REAL(*)[NY_GLOBAL])malloc(sizeof(REAL) * NY_GLOBAL * NZ_GLOBAL);

	memset((void*)U2d1, 0, NY_GLOBAL * NZ_GLOBAL * sizeof(REAL));

	if (my_id == 0){
		U01 = (REAL(*)[NY_GLOBAL])malloc(sizeof(REAL) * NY_GLOBAL * NZ_GLOBAL);
	}
	//--------------------------------
	get_i_node(ia, &node_i, &i_local);

	int j, k;
	if(npx == node_i){
		for (k = LAP; k < nz + LAP; k++)
		{
			for (j = LAP; j < ny + LAP; j++)
			{
				U2d1[k - LAP + k_offset[npz]][j - LAP + j_offset[npy]] = U1[k][j][i_local + LAP];
			}
		}
	}

	MPI_Reduce(&U2d1[0][0], &U01[0][0], NY_GLOBAL * NZ_GLOBAL, OCFD_DATA_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
//	if (my_id == 0)
//		FWRITE(U0, sizeof(REAL), NY_GLOBAL * NZ_GLOBAL, file)

	if(my_id == 0) FWRITE(U01, sizeof(REAL), NY_GLOBAL * NZ_GLOBAL, file)

	free(U2d1);
	if (my_id == 0){
		free(U01);
	}
}


void write_2d_XZa(
	FILE *file,
	int ja,
	REAL *pU1)
{

	REAL(*U1)[ny + 2*LAP][nx + 2*LAP] = (REAL(*)[ny + 2*LAP][nx + 2*LAP])(pU1);
	REAL(*U2d1)[NX_GLOBAL], (*U01)[NX_GLOBAL];
	int node_j, j_local;

	U2d1 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NZ_GLOBAL);

	memset((void*)U2d1, 0, NX_GLOBAL * NZ_GLOBAL * sizeof(REAL));

	if (my_id == 0){
		U01 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NZ_GLOBAL);
	}
	//--------------------------------
	get_j_node(ja, &node_j, &j_local);

	int i, k;
	if(npy == node_j){
		for (k = LAP; k < nz + LAP; k++)
		{
			for (i = LAP; i < nx + LAP; i++)
			{
				U2d1[k - LAP + k_offset[npz]][i - LAP + i_offset[npx]] = U1[k][j_local + LAP][i];
			}
		}
	}

	MPI_Reduce(&U2d1[0][0], &U01[0][0], NX_GLOBAL * NZ_GLOBAL, OCFD_DATA_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
//	if (my_id == 0)
//		FWRITE(U0, sizeof(REAL), NY_GLOBAL * NZ_GLOBAL, file)

	if(my_id == 0) FWRITE(U01, sizeof(REAL), NX_GLOBAL * NZ_GLOBAL, file)

	free(U2d1);
	if (my_id == 0){
		free(U01);
	}
}


//--------------------------------------------------------------
//-----Write a 2D Y-Z (j-k) plane from 3-D array
//void write_2d_YZa(
//	FILE *file,
//	int ia,
//	REAL *pU)
//{
//
//	REAL(*U)
//	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pU, nx + 2 * LAP, ny + 2 * LAP);
//	REAL(*U2d), (*U0);
//	int node_i, i_local;
//
//	U2d = (REAL *)malloc(sizeof(REAL) * ny * nz);
//	if (my_id == 0)
//		U0 = (REAL *)malloc(sizeof(REAL) * NY_GLOBAL * NZ_GLOBAL);
//	//--------------------------------
//	get_i_node(ia, &node_i, &i_local);
//	i_local += LAP;
//	int k, j;
//	REAL *tmp = U2d;
//	for (k = LAP; k < nz + LAP; k++)
//	{
//		for (j = LAP; j < ny + LAP; j++)
//		{
//			(*tmp++) = U[k][j][i_local];
//		}
//	}
//
//	for (int proc_k = 0; proc_k < NPZ0; k++)
//	{
//		for (int kk = k_offset[proc_k]; kk < k_offset[proc_k] + k_nn[proc_k]; kk++)
//		{
//			for (int proc_j = 0; proc_j < NPY0; proc_j++)
//			{
//				if (npx == node_i && npy == proc_j && npz == proc_k)
//				{
//					k = kk - k_offset[proc_k];
//					MPI_Bsend(U2d + k * ny, ny, OCFD_DATA_TYPE, 0, kk, MPI_COMM_WORLD);
//				}
//				if (my_id == 0)
//				{
//					int recv_offset = j_offset[proc_j] + NY_GLOBAL * kk;
//					MPI_Status status;
//					MPI_Recv(U0 + recv_offset, j_nn[proc_j], OCFD_DATA_TYPE, PROCIdx2Num(node_i, proc_j, proc_k), kk, MPI_COMM_WORLD, &status);
//				}
//			}
//			MPI_Barrier(MPI_COMM_WORLD);
//		}
//	}
//	if (my_id == 0)
//		FWRITE(U0, sizeof(REAL), NY_GLOBAL * NZ_GLOBAL, file)
//
//	free(U2d);
//	if (my_id == 0)
//		free(U0);
//}

//-------------------------------------------------
//----Write a 2d xz-plane from 3d array------------------------

//void write_2d_XZa(
//	FILE *file,
//	int ja,
//	REAL *pU)
//{
//
//	REAL(*U)
//	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pU, nx + 2 * LAP, ny + 2 * LAP);
//	REAL(*U2d), (*U0);
//	int node_j, j_local;
//
//	U2d = (REAL *)malloc(sizeof(REAL) * nx * nz);
//	if (my_id == 0)
//		U0 = (REAL *)malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);
//	//--------------------------------
//	get_j_node(ja, &node_j, &j_local);
//	j_local += LAP;
//	int k, i;
//	REAL *tmp = U2d;
//	for (k = LAP; k < nz + LAP; k++)
//	{
//		for (i = LAP; i < nx + LAP; i++)
//		{
//			(*tmp++) = U[k][j_local][i];
//		}
//	}
//	for (int proc_k = 0; proc_k < NPZ0; k++)
//	{
//		for (int kk = k_offset[proc_k]; kk < k_offset[proc_k] + k_nn[proc_k]; kk++)
//		{
//			for (int proc_i = 0; proc_i < NPX0; proc_i++)
//			{
//				if (npy == node_j && npx == proc_i && npz == proc_k)
//				{
//					k = kk - k_offset[proc_k];
//					MPI_Bsend(U2d + k * nx, nx, OCFD_DATA_TYPE, 0, kk, MPI_COMM_WORLD);
//				}
//				if (my_id == 0)
//				{
//					int recv_offset = i_offset[proc_i] + NX_GLOBAL * kk;
//					MPI_Status status;
//					MPI_Recv(U0 + recv_offset, i_nn[proc_i], OCFD_DATA_TYPE, PROCIdx2Num(proc_i, node_j, proc_k), kk, MPI_COMM_WORLD, &status);
//				}
//			}
//			MPI_Barrier(MPI_COMM_WORLD);
//		}
//	}
//	if (my_id == 0)
//		FWRITE(U0, sizeof(REAL), NX_GLOBAL * NZ_GLOBAL, file)
//
//	free(U2d);
//	if (my_id == 0)
//		free(U0);
//}
//--------------------------------------------------

//----Write points from 3d array------------------------
// 需要明确外界输入文件中，ia，ja，ka所使用的下标体系
void write_points(
	FILE *file,
	REAL *pU,
	int mpoints,
	int *ia,
	int *ja,
	int *ka)
{
	int node_i, node_j, node_k, i_local, j_local, k_local;
	REAL(*U)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pU, nx + 2 * LAP, ny + 2 * LAP);
	REAL *U1;
	U1 = (REAL *)malloc(sizeof(REAL) * mpoints);
	//--------------------------------
	for (int m = 0; m < mpoints; m++)
	{
		get_i_node(ia[m], &node_i, &i_local);
		get_j_node(ja[m], &node_j, &j_local);
		get_k_node(ka[m], &node_k, &k_local);
		if (npx == node_i && npy == node_j && npz == node_k)
		{
			MPI_Bsend(&U[k_local + LAP][j_local + LAP][i_local + LAP], 1, OCFD_DATA_TYPE, 0, 0, MPI_COMM_WORLD);
		}
		if (my_id == 0)
		{
			MPI_Status status;
			MPI_Recv(&U1[m], 1, OCFD_DATA_TYPE, PROCIdx2Num(node_i, node_j, node_k), 0, MPI_COMM_WORLD, &status);
		}
	}
	if (my_id == 0)
		FWRITE(U1, sizeof(REAL), mpoints, file)
	free(U1);
}

//--------------------------------------------------

//--------------------------------------------------
//void read_3d(
//	FILE *file,
//	REAL *pU)
//{
//
//	REAL(*U)
//	[ny][nx] = PTR2ARRAY2(pU, nx, ny);
//
//	REAL(*buff2d)
//	[NX_GLOBAL], (*buff1)[NX_GLOBAL], *buff2, *buff_recv;
//	int sendcounts1[NPY0], displs1[NPY0], sendcounts2[NPX0], displs2[NPX0];
//	//---------------------------------------------------------------
//	if (npx == 0)
//	{
//		if (npy == 0)
//		{
//			buff2d = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);
//		}
//		buff1 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * ny);
//		buff2 = (REAL *)malloc(sizeof(REAL) * NX_GLOBAL * ny);//NY_GLOBAL > ny
//	}
//	buff_recv = (REAL *)malloc(sizeof(REAL) * nx * ny);
//
//	if (my_id == 0)
//		printf("read 3d data ...\n");
//	// sendcounts1 displs1用于j方向分布
//	for (int j = 0; j < NPY0; j++)
//	{
//		sendcounts1[j] = NX_GLOBAL * j_nn[j];
//		displs1[j] = j_offset[j] * NX_GLOBAL;
//	}
//
//	for (int i = 0; i < NPX0; i++)
//	{
//		sendcounts2[i] = ny * i_nn[i];
//		displs2[i] = i_offset[i] * ny;
//	}
//
//	int proc_k, k_local;
//	for (int kk = 0; kk < NZ_GLOBAL; kk++)
//	{
//		get_k_node(kk, &proc_k, &k_local);
//		if (my_id == 0)
//			FREAD(buff2d, sizeof(REAL), NX_GLOBAL * NY_GLOBAL, file)
//
//		if (proc_k != 0)
//		{
//			// k方向发送
//			MPI_Status status;
//			if (my_id == 0)
//				MPI_Send(buff2d, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, proc_k * (NPX0 * NPY0), 6666, MPI_COMM_WORLD);
//			if (my_id == proc_k * NPX0 * NPY0)
//				MPI_Recv(buff2d, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, 0, 6666, MPI_COMM_WORLD, &status);
//		}
//		if (npz == proc_k)
//		{
//			// j方向分散
//			if (npx == 0)
//			{
//				MPI_Scatterv(buff2d, sendcounts1, displs1, OCFD_DATA_TYPE, buff1, NX_GLOBAL * ny, OCFD_DATA_TYPE, 0, MPI_COMM_Y);
//
//				REAL *pbuff_recv;
//				REAL *ppU;
//				// i方向数据准备与离散
//				for (int npx1 = 0; npx1 < NPX0; npx1++)
//				{
//					ppU = buff2 + displs2[npx1];
//					for (int j = 0; j < ny; j++)
//					{
//						for (int i = i_offset[npx1]; i < i_offset[npx1] + i_nn[npx1]; i++)
//						{
//							(*ppU++) = buff1[j][i];
//						}
//					}
//				}
//			}
//			//buff_recv = buff2;
//			MPI_Scatterv(buff2, sendcounts2, displs2, OCFD_DATA_TYPE, buff_recv, nx * ny, OCFD_DATA_TYPE, 0, MPI_COMM_X);
//
//			// 数据分布
//			{
//				REAL *pbuff_recv;
//				REAL *ppU;
//				ppU = pU + k_local * nx * ny;
//				pbuff_recv = buff_recv;
//				for (int nn = 0; nn < nx * ny; nn++)
//				{
//					(*ppU++) = (*pbuff_recv++);
//				}
//			}
//		}
//	}
//
//	if (npx == 0)
//	{
//		if (npy == 0)
//		{
//			free(buff2d);
//		}
//		free(buff1);
//		free(buff2);
//	}
//	free(buff_recv);
//}

//void read_3d(
//	MPI_File file,
//	REAL *pU)
//{
//	size_t displs_start, displs_end, displs_k_start, displs_k_end;
//
//    REAL *buff_recv = (REAL *)malloc(sizeof(REAL) * NX_GLOBAL * ny);
//
//    displs_start = (2*sizeof(int) + NX_GLOBAL*NY_GLOBAL*sizeof(REAL)) * k_offset[npz];
//    displs_end = (2*sizeof(int) + NX_GLOBAL*NY_GLOBAL*sizeof(REAL)) * (NZ_GLOBAL-k_offset[npz]-nz);
//
//	displs_k_start = sizeof(int) + (i_offset[npx] + j_offset[npy] * NX_GLOBAL) * sizeof(REAL);
//	displs_k_end = 2*sizeof(int) + (NY_GLOBAL - ny) * NX_GLOBAL * sizeof(REAL);
//
//    if (my_id == 0) printf("read 3d data ...\n");
//
//    MPI_File_seek(file, displs_start + displs_k_start, MPI_SEEK_CUR);
//
//    for(int k=0; k<nz; k++){
//
//        MPI_File_read(file, buff_recv, NX_GLOBAL*ny, OCFD_DATA_TYPE, &status);
//
//        MPI_File_seek(file, displs_k_end, MPI_SEEK_CUR);
//
//    // 数据分布
//       {
//           REAL *ppU;
//       	   ppU = pU + k * nx * ny;
//           
//           for(int j=0;j<ny;j++){
//               for(int i=0;i<nx;i++){
//           	     *(ppU+j*nx+i) = *(buff_recv+j*NX_GLOBAL+i);
//               }
//           }
//       }
//    }
//
//    MPI_File_seek(file, displs_end - displs_k_start, MPI_SEEK_CUR);		 
//
//    free(buff_recv);
//}

void read_3d(
	MPI_File file,
	MPI_Offset offset,
	REAL *pU)
{
	size_t displs_start, displs_k_start;

    REAL *buff_recv = (REAL *)malloc(sizeof(REAL) * NX_GLOBAL * ny);

    displs_start = (2*sizeof(int) + NX_GLOBAL*NY_GLOBAL*sizeof(REAL)) * k_offset[npz];

	displs_k_start = sizeof(int) + (i_offset[npx] + j_offset[npy] * NX_GLOBAL) * sizeof(REAL);

	displs_start += displs_k_start + offset;

    if (my_id == 0) printf("read 3d data ...\n");

    for(int k=0; k<nz; k++){

        MPI_File_read_at(file, displs_start, buff_recv, NX_GLOBAL*ny, OCFD_DATA_TYPE, &status);

		displs_start += 2*sizeof(int) + NY_GLOBAL * NX_GLOBAL * sizeof(REAL);


    // 数据分布
       {
           REAL *ppU;
       	   ppU = pU + k * nx * ny;
           
           for(int j=0;j<ny;j++){
               for(int i=0;i<nx;i++){
           	     *(ppU+j*nx+i) = *(buff_recv+j*NX_GLOBAL+i);
               }
           }
       }
    }	 

    free(buff_recv);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

void write_3d(
	MPI_File file,
	MPI_Offset offset,
	REAL *pU)
{

	REAL(*U)[ny][nx] = PTR2ARRAY2(pU, nx, ny);
	REAL(*buff1)[NX_GLOBAL], *buff2, *buff_send;
	int *buff2d;

	size_t size = NX_GLOBAL*NY_GLOBAL*sizeof(REAL);
	size_t displs_k;

	int recvcounts1[NPY0], displs1[NPY0], recvcounts2[NPX0], displs2[NPX0];


	displs_k = k_offset[npz] * (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) + offset;

	if (npx == 0)
	{
		if (npy == 0)
		{
			buff2d = (int*)malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL + sizeof(int) * 2);
			*buff2d = size;
			*(buff2d + 1 + NX_GLOBAL * NY_GLOBAL * 2) = size;
		}
		buff1 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * ny);
		buff2 = (REAL *)malloc(sizeof(REAL) * NX_GLOBAL * ny);
	}
	buff_send = (REAL *)malloc(sizeof(REAL) * nx * ny);

	//---------------------------------------------------------------
	if (my_id == 0)
		printf("write 3d data ...\n");
	// recvconts1 ， displs1 存储j方向收集时所使用的个数与偏移，
	// 由于j方向收集发生在i方向收集之后，因此只有一列参与j方向收集
	for (int j = 0; j < NPY0; j++)
	{
		recvcounts1[j] = NX_GLOBAL * j_nn[j];
		displs1[j] = j_offset[j] * NX_GLOBAL;
	}
	// i方向收集所需偏移与数量
	for (int i = 0; i < NPX0; i++)
	{
		recvcounts2[i] = ny * i_nn[i];
		displs2[i] = i_offset[i] * ny;
	}

	// 按数据的k面进行循环	
	for (int kk = 0; kk < nz; kk++)
	{
		REAL *pbuff_send = (REAL *)buff_send;
		REAL *ppU = pU + kk * nx * ny;
		// i方向收集数据准备
		for (int n = 0; n < nx * ny; n++)
			(*pbuff_send++) = (*ppU++);

		MPI_Gatherv(buff_send, nx * ny, OCFD_DATA_TYPE, buff2, recvcounts2, displs2, OCFD_DATA_TYPE, 0, MPI_COMM_X);

		if (npx == 0)
		{
			// j方向收集数据调序
			for (int npx1 = 0; npx1 < NPX0; npx1++)
			{
				ppU = buff2 + displs2[npx1];

				for (int j = 0; j < ny; j++)
				{
					for (int i = i_offset[npx1]; i < i_offset[npx1] + i_nn[npx1]; i++)
					{
						buff1[j][i] = (*ppU++);
					}
				}
			}
			MPI_Gatherv(buff1, NX_GLOBAL * ny, OCFD_DATA_TYPE, (REAL*)(buff2d + 1), recvcounts1, displs1, OCFD_DATA_TYPE, 0, MPI_COMM_Y);
		}
		

        if (npx == 0 && npy == 0){
            MPI_File_write_at(file, displs_k, buff2d, 2*(NX_GLOBAL*NY_GLOBAL+1), MPI_INT, &status);
		}

		displs_k += 2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL);
	}

	if (npx == 0)
	{
		if (npy == 0)
		{
			free(buff2d);
		}
		free(buff1);
		free(buff2);
	}
	free(buff_send);
}


//void write_3d(
//	MPI_File file,
//	REAL *pU)
//{
//    size_t displs_xy;
//	size_t size = NX_GLOBAL*NY_GLOBAL*sizeof(REAL);
//    size_t displs_non0_start = (2*sizeof(int) + NX_GLOBAL*NY_GLOBAL*sizeof(REAL)) * k_offset[npz];
//	size_t displs_non0_end = (2*sizeof(int) + NX_GLOBAL*NY_GLOBAL*sizeof(REAL))*(NZ_GLOBAL-k_offset[npz]-nz);
//
//    REAL *buff_recv = (REAL *)malloc(sizeof(REAL) * nx * ny);
//    displs_xy = (i_offset[npx] + j_offset[npy] * NX_GLOBAL) * sizeof(REAL);
//
//    if(my_id == 0){
//        for(int i=0; i<k_offset[npz]; i++){
//            MPI_File_write_all(file, &size, 1, MPI_INT, &status);
//            MPI_File_seek(file, size, MPI_SEEK_CUR);		
//            MPI_File_write_all(file, &size, 1, MPI_INT, &status);
//        }
//    }else{
//        MPI_File_seek(file, displs_non0_start, MPI_SEEK_CUR);		
//    }
//
//    for(int k=0; k<nz; k++){
//    // 数据分布
//       {
//           REAL *ppU;
//       	   ppU = pU + k * nx * ny;
//           
//           for(int j=0;j<ny;j++){
//               for(int i=0;i<nx;i++){
//		           *(buff_recv+j*nx+i) = *(ppU+j*nx+i);
//               }
//           }
//       }
//
//        if(my_id == 0){
//	        MPI_File_write_all(file, &size, 1,  MPI_INT, &status);
//	    }else{
//            MPI_File_seek(file, sizeof(int), MPI_SEEK_CUR);
//	    }
//
//        MPI_File_seek(file, displs_xy, MPI_SEEK_CUR);
//
//        for(int j = 0; j < ny; j++){
//            MPI_File_write_all(file, buff_recv + nx*j, nx, OCFD_DATA_TYPE, &status);
//
//            MPI_File_seek(file, sizeof(REAL)*(NX_GLOBAL-nx), MPI_SEEK_CUR);
//        }
//
//        MPI_File_seek(file, sizeof(REAL)*((NY_GLOBAL-j_offset[npy]-ny)*NX_GLOBAL-i_offset[npx]), MPI_SEEK_CUR);
//       
//       	if(my_id == 0){
//	        MPI_File_write_all(file, &size, 1,  MPI_INT, &status);
//	    }else{
//            MPI_File_seek(file, sizeof(int), MPI_SEEK_CUR);
//	    }
//
//   }
//
//    if(my_id == 0){
//        for(int i=0; i<(NZ_GLOBAL-k_offset[npz]-nz); i++){
//	        MPI_File_write_all(file, &size, 1,  MPI_INT, &status);
//            MPI_File_seek(file, size, MPI_SEEK_CUR);		
//	        MPI_File_write_all(file, &size, 1,  MPI_INT, &status);
//        }
//    }else{
//        MPI_File_seek(file, displs_non0_end, MPI_SEEK_CUR);		
//    }
//
//   if (my_id == 0) printf("write 3d data ...\n");
//
//   free(buff_recv);
//}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//void write_3d(
//	FILE *file,
//	REAL *pU)
//{
//
//	REAL(*U)
//	[ny][nx] = PTR2ARRAY2(pU, nx, ny);
//	REAL(*buff2d)
//	[NX_GLOBAL], (*buff1)[NX_GLOBAL], *buff2, *buff_send;
//
//	int recvcounts1[NPY0], displs1[NPY0], recvcounts2[NPX0], displs2[NPX0];
//
//	if (npx == 0)
//	{
//		if (npy == 0)
//		{
//			buff2d = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * NY_GLOBAL);
//		}
//		buff1 = (REAL(*)[NX_GLOBAL])malloc(sizeof(REAL) * NX_GLOBAL * ny);
//		buff2 = (REAL *)malloc(sizeof(REAL) * NX_GLOBAL * ny);
//	}
//	buff_send = (REAL *)malloc(sizeof(REAL) * nx * ny);
//
//	//---------------------------------------------------------------
//	if (my_id == 0)
//		printf("write 3d data ...\n");
//	// recvconts1 ， displs1 存储j方向收集时所使用的个数与偏移，
//	// 由于j方向收集发生在i方向收集之后，因此只有一列参与j方向收集
//	for (int j = 0; j < NPY0; j++)
//	{
//		recvcounts1[j] = NX_GLOBAL * j_nn[j];
//		displs1[j] = j_offset[j] * NX_GLOBAL;
//	}
//	// i方向收集所需偏移与数量
//	for (int i = 0; i < NPX0; i++)
//	{
//		recvcounts2[i] = ny * i_nn[i];
//		displs2[i] = i_offset[i] * ny;
//	}
//
//	// 按数据的k面进行循环
//	int proc_k, k_local;
//	for (int kk = 0; kk < NZ_GLOBAL; kk++)
//	{
//		get_k_node(kk, &proc_k, &k_local);
//		if (npz == proc_k)
//		{
//			REAL *pbuff_send = (REAL *)buff_send;
//			REAL *ppU = pU + k_local * nx * ny;
//			// i方向收集数据准备
//			for (int n = 0; n < nx * ny; n++)
//				(*pbuff_send++) = (*ppU++);
//			MPI_Gatherv(buff_send, nx * ny, OCFD_DATA_TYPE, buff2, recvcounts2, displs2, OCFD_DATA_TYPE, 0, MPI_COMM_X);
//
//			if (npx == 0)
//			{
//				// j方向收集数据调序
//				for (int npx1 = 0; npx1 < NPX0; npx1++)
//				{
//					ppU = buff2 + displs2[npx1];
//
//					for (int j = 0; j < ny; j++)
//					{
//						for (int i = i_offset[npx1]; i < i_offset[npx1] + i_nn[npx1]; i++)
//						{
//							buff1[j][i] = (*ppU++);
//						}
//					}
//				}
//				MPI_Gatherv(buff1, NX_GLOBAL * ny, OCFD_DATA_TYPE, buff2d, recvcounts1, displs1, OCFD_DATA_TYPE, 0, MPI_COMM_Y);
//			}
//		}
//
//		//
//
//		// k 方向收集
//		if (proc_k != 0)
//		{
//			if (npx == 0 && npy == 0 && npz == proc_k)
//			{
//				MPI_Send(buff2d, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, 0, 666, MPI_COMM_WORLD);
//			}
//			if (my_id == 0)
//			{
//				MPI_Status status;
//				MPI_Recv(buff2d, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, proc_k * NPX0 * NPY0, 666, MPI_COMM_WORLD, &status);
//			}
//		}
//		if (my_id == 0)
//			FWRITE(buff2d, sizeof(REAL), NX_GLOBAL * NY_GLOBAL, file)
//	}
//
//	if (npx == 0)
//	{
//		if (npy == 0)
//		{
//			free(buff2d);
//		}
//		free(buff1);
//		free(buff2);
//	}
//	free(buff_send);
//}

//------------------------------------------------------------------------------------------------------------------
//------------------------------------Write blockdata from 3d array-------------------------------------------------

void write_blockdata(
	FILE *file,
	REAL *pU,
	int ib,
	int ie,
	int jb,
	int je,
	int kb,
	int ke)
{
	int nx1 = ie - ib + 1, ny1 = je - jb + 1, nz1 = ke - kb + 1;
	int i, j, k, i0, j0, k0, i1, j1, k1;

	REAL(*U)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pU, nx + 2 * LAP, ny + 2 * LAP);
	REAL U1[nz1][ny1][nx1], U0[nz1][ny1][nx1];
	//--------------------------------
	REAL *p = &U1[0][0][0];
	for (int i = 0; i < nx1 * ny1 * nz1; i++)
	{
		(*p++) = 0.0;
	}
	p = &U0[0][0][0];
	for (int i = 0; i < nx1 * ny1 * nz1; i++)
	{
		(*p++) = 0.0;
	}

	// 假设in文件使用fortran下标 , 从1开始
	ib -= 1;
	jb -= 1;
	kb -= 1;

	int gkb = k_offset[npz];
	int gjb = j_offset[npy];
	int gib = i_offset[npx];

	for (k = 0; k < nz; k++)
	{
		k0 = k + gkb;
		if (!(k0 >= kb && k0 < ke))
			continue;
		k1 = k0 - kb;
		for (j = 0; j < ny; j++)
		{
			j0 = j + gjb;
			if (!(j0 >= jb && j0 < je))
				continue;
			j1 = j0 - jb;
			for (i = 0; i < nx; i++)
			{
				i0 = i + gib;
				if (!(i0 >= ib && i0 < ie))
					continue;
				i1 = i0 - ib;
				U1[k1][j1][i1] = U[k + LAP][j + LAP][i + LAP];
			}
		}
	}
	MPI_Reduce(&U1[0][0][0], &U0[0][0][0], nx1 * ny1 * nz1, OCFD_DATA_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (my_id == 0)
		FWRITE(&U0[0][0][0], sizeof(REAL), nx1 * ny1 * nz1, file)
}
#ifdef __cplusplus
}
#endif
//--------------------------------------------------

//Read & save file

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "mpi.h"

#include "OCFD_ana.h"
#include "parameters.h"
#include "utility.h"
#include "OCFD_IO.h"
#include "OCFD_IO_mpi.h"
#include "io_warp.h"

#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "commen_kernel.h"

#ifdef __cplusplus
extern "C"{
#endif
void read_file(
	int Iflag_av,
	REAL * pd,
	REAL * pu,
	REAL * pv,
	REAL * pw,
	REAL * pT)
{

	// Iflag_av == 0 , read opencfd data file; ==1, read averaged file
	int Irestart_step;
	char filename1[100];
	//-----------------------------------------------------------
    if(Iflag_av == 0){
	    Irestart_step = -1;
	    if (my_id == 0)
	    {
	    	FILE *tmp_file;
	    	if ((tmp_file = fopen("Opencfd.msg", "r")))
	    	{
	    		fread(&Irestart_step, sizeof(int), 1, tmp_file);
	    		fclose(tmp_file);
	    	}
	    	else
	    	{
	    		printf("Opencfd.msg is not exist, read initial file : opencfd.dat ......\n");
	    	}
	    }
    
	    MPI_Bcast(&Irestart_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    if (Irestart_step < 0)
	    {
	    	sprintf(filename1, "opencfd.dat");
	    }
	    else
	    {
	    	sprintf(filename1, "OCFD%08d.dat", Irestart_step);
	    }
	    MPI_File tmp_file;
	    int tmp[3];
	    
	    if(my_id == 0) printf("read initial data file: %s \n\n", filename1);
	    MPI_File_open(MPI_COMM_WORLD, filename1, MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);
        
	    MPI_File_read_at_all(tmp_file, sizeof(int), tmp, 1, MPI_INT, &status);
        MPI_File_read_at_all(tmp_file, 2*sizeof(int), tmp+1, 1, OCFD_DATA_TYPE, &status);

		MPI_Offset offset = 3*sizeof(int)+sizeof(REAL);
	
	    Istep = tmp[0];
	    tt = *(REAL*)(tmp+1);
	    if(my_id == 0) printf("Istep=%d , tt=%lf\n", Istep, tt);

	    read_3d1(tmp_file, offset, pd);
		offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	    read_3d1(tmp_file, offset, pu);
		offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	    read_3d1(tmp_file, offset, pv);
		offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	    read_3d1(tmp_file, offset, pw);
		offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	    read_3d1(tmp_file, offset, pT);

        MPI_File_close(&tmp_file);
	    //------------------------
	    if (my_id == 0)
	    	printf("read data ok\n");
    }

	//--------------------
	if(Iflag_av == 1)
	{
	    // averaged file
	    	//char *tmp_char = strstr(filename1, ".dat");

		sprintf(filename1, "opencfd.average");

		if (access(filename1, F_OK) == -1){ 

			//The file not exist
      		if(my_id == 0) printf("Average file: %s is not exit\n\n", filename1);
      		Istep_average = 0;
      		tt_average = 0.0;

            init_time_average();
     	}else{
	        if (my_id == 0)
				printf("read average_data begin\n");

			MPI_File tmp_file;
			MPI_File_open(MPI_COMM_WORLD, filename1, MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);  
            int tmp[3];
	    
			MPI_File_read_at_all(tmp_file, sizeof(int), tmp, 1, MPI_INT, &status);
        	MPI_File_read_at_all(tmp_file, 2*sizeof(int), tmp+1, 1, OCFD_DATA_TYPE, &status);

			MPI_Offset offset = 3*sizeof(int)+sizeof(REAL);

            Istep_average = tmp[0];
            tt_average = *(REAL*)(tmp+1);
            if(my_id == 0) printf("Istep_average=%d , tt_average=%lf\n", Istep_average, tt_average);

			init_time_average();

            read_3d1(tmp_file, offset, pdm);
			offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
            read_3d1(tmp_file, offset, pum);
			offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
            read_3d1(tmp_file, offset, pvm);
			offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
            read_3d1(tmp_file, offset, pwm);
			offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
            read_3d1(tmp_file, offset, pTm);

            MPI_File_close(&tmp_file);
    
	//--    ----------------------
	        if (my_id == 0)
				printf("read average_data ok\n");
				
			memcpy_inner(pdm , pdm_d->ptr , pdm_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
			memcpy_inner(pum , pum_d->ptr , pum_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
			memcpy_inner(pvm , pvm_d->ptr , pvm_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
			memcpy_inner(pwm , pwm_d->ptr , pwm_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
			memcpy_inner(pTm , pTm_d->ptr , pTm_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
		}
	    average_IO = 0;
	}
	//---------------------
}
//----------------------------------------------------------------------------------

//================================================================================
void OCFD_save_final(
	int Iflag_av,
	int Istep_name,
	REAL * pd,
	REAL * pu,
	REAL * pv,
	REAL * pw,
	REAL * pT,
	REAL *O2, 
	REAL *N2)
{
							    
	// Iflag_av==0, write opencfd file; ==1, write averaged data file

	char filename1[120];
	//-------------------------------------------
	MPI_File tmp_file;
	int tmp[3];
	int size_tmp = sizeof(tmp);

        if(Iflag_av == 0){
            sprintf(filename1, "OCFD%08d.dat", Istep_name);
		}else{
		    sprintf(filename1, "OCFD%08d.average", Istep_name);
		}
        if(my_id == 0) printf("write data file: %s\n", filename1);
	
	MPI_File_open(MPI_COMM_WORLD, filename1, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

	if(Iflag_av == 0){
            
	    tmp[0] = Istep;
        *(REAL*)(tmp + 1) = tt;

	    MPI_File_write_at_all(tmp_file, 0, &size_tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, sizeof(int), tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int), tmp+1, 1, OCFD_DATA_TYPE, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int)+sizeof(REAL), &size_tmp, 1, MPI_INT, &status);
	}else{
            
	    tmp[0] = Istep_average;
        *(REAL*)(tmp + 1) = tt_average;
	    
	    MPI_File_write_at_all(tmp_file, 0, &size_tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, sizeof(int), tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int), tmp+1, 1, OCFD_DATA_TYPE, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int)+sizeof(REAL), &size_tmp, 1, MPI_INT, &status);
	}

	MPI_Offset offset = 3*sizeof(int)+sizeof(REAL);

	write_3d1(tmp_file, offset, pd);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pu);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pv);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pw);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pT);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, O2);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, N2);

    MPI_File_close(&tmp_file);

	//if (my_id == 0)
	//{
	//	if (Iflag_av == 0)
	//	{
	//		printf("write data OK\n");
	//		tmp_file = fopen("Opencfd.msg", "a");
	//		fprintf(tmp_file, "%d", Istep_name);
	//		fclose(tmp_file);
	//	}
	//}
}

void OCFD_save(
	int Iflag_av,
	int Istep_name,
	REAL * pd,
	REAL * pu,
	REAL * pv,
	REAL * pw,
	REAL * pT)
{
							    
	// Iflag_av==0, write opencfd file; ==1, write averaged data file

	char filename1[120];
	//-------------------------------------------
	MPI_File tmp_file;
	int tmp[3];
	int size_tmp = sizeof(tmp);

        if(Iflag_av == 0){
            sprintf(filename1, "OCFD%08d.dat", Istep_name);
		}else{
		    sprintf(filename1, "OCFD%08d.average", Istep_name);
		}
        if(my_id == 0) printf("write data file: %s\n", filename1);
	
	MPI_File_open(MPI_COMM_WORLD, filename1, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

	if(Iflag_av == 0){
            
	    tmp[0] = Istep;
        *(REAL*)(tmp + 1) = tt;

	    MPI_File_write_at_all(tmp_file, 0, &size_tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, sizeof(int), tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int), tmp+1, 1, OCFD_DATA_TYPE, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int)+sizeof(REAL), &size_tmp, 1, MPI_INT, &status);
	}else{
            
	    tmp[0] = Istep_average;
        *(REAL*)(tmp + 1) = tt_average;
	    
	    MPI_File_write_at_all(tmp_file, 0, &size_tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, sizeof(int), tmp, 1, MPI_INT, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int), tmp+1, 1, OCFD_DATA_TYPE, &status);
	    MPI_File_write_at_all(tmp_file, 2*sizeof(int)+sizeof(REAL), &size_tmp, 1, MPI_INT, &status);
	}

	MPI_Offset offset = 3*sizeof(int)+sizeof(REAL);

	write_3d1(tmp_file, offset, pd);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pu);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pv);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pw);
	offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
	write_3d1(tmp_file, offset, pT);

    MPI_File_close(&tmp_file);

	//if (my_id == 0)
	//{
	//	if (Iflag_av == 0)
	//	{
	//		printf("write data OK\n");
	//		tmp_file = fopen("Opencfd.msg", "a");
	//		fprintf(tmp_file, "%d", Istep_name);
	//		fclose(tmp_file);
	//	}
	//}
}
//-------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------
void write_3d1(
	MPI_File file,
	MPI_Offset offset,
	REAL *pU)
{
	int i, j, k;
	REAL(*U)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pU, nx + 2 * LAP, ny + 2 * LAP);
	REAL(*U1)
	[ny][nx] = (REAL(*)[ny][nx])malloc(nx * ny * nz * sizeof(REAL));
	REAL *pU1 = (REAL*)U1;
	for (k = LAP; k < nz + LAP; k++)
	{
		for (j = LAP; j < ny + LAP; j++)
		{
			for (i = LAP; i < nx + LAP; i++)
			{
				(*pU1++) = U[k][j][i];
			}
		}
	}
	pU1 = &(U1[0][0][0]);
	write_3d(file, offset, pU1);
	free(U1);
}

void read_3d1(
	MPI_File file,
	MPI_Offset offset,
	REAL *pU)
{

	int i, j, k;
	REAL(*U1)
	[ny][nx] = (REAL(*)[ny][nx])malloc(nx * ny * nz * sizeof(REAL));
	REAL(*U)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pU, nx + 2 * LAP, ny + 2 * LAP);
	REAL *pU1 = (REAL*)U1;
	read_3d(file, offset, pU1);
	for (k = LAP; k < nz + LAP; k++)
	{
		for (j = LAP; j < ny + LAP; j++)
		{
			for (i = LAP; i < nx + LAP; i++)
			{
				U[k][j][i] = (*pU1++);
			}
		}
	}
	free(U1);
}
#ifdef __cplusplus
}
#endif

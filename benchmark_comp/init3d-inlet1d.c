#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE  *fp;
MPI_File tmp_file;
MPI_Status status;

char str[100];

int _DOU = sizeof(double);
int my_id, n_processe;
int nx, ny, nz, N, NZ, *NPZ, *NP;
int nx_inlet, nx_flat, nx_conner_in, nx_conner_out, nx_reattach, nx_buff;
int ny_nearwall, ny_outer;

double *d1, *u1, *v1, *T1, *yy;
double *d2, *u2, *v2, *T2, *w2;

void mpi_init(int *Argc, char ***Argv);
void Read_parameter();
void Read_flow1d_inlet();
void Creat_flow2d();
void Output_flow3d();
void Finalize();
void output1d(int n, double *xx);
void output2d(int nx, int ny, double *xx);

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();

    Read_flow1d_inlet();

    Creat_flow2d();

    Output_flow3d();

    Finalize();

    return 0;
}

void mpi_init(int *Argc , char *** Argv){

	MPI_Init(Argc, Argv);

    MPI_Comm_rank(MPI_COMM_WORLD , &my_id);
    
    MPI_Comm_size(MPI_COMM_WORLD, &n_processe);

    N = n_processe;
}

void Read_parameter(){
    if(my_id == 0){
        if((fp = fopen("swept-compression-grid.in", "r")) == NULL){
            printf("Can't open this file: 'swept-compression-grid.in'\n");
            exit(0);
        }
    
        fgets(str, 100, fp);
        fscanf(fp, "%d%d%d%d%d%d\n", &nx_inlet, &nx_flat, &nx_conner_in, &nx_conner_out, &nx_reattach, &nx_buff);

        nx = nx_inlet + nx_flat + nx_conner_in + nx_conner_out + nx_reattach + nx_buff;

        fgets(str, 100, fp);
        fgets(str, 100, fp);

        fgets(str, 100, fp);
        fscanf(fp, "%d%d\n", &ny_nearwall, &ny_outer);

        ny = ny_nearwall + ny_outer;

        fgets(str, 100, fp);
        fgets(str, 100, fp);

        fgets(str, 100, fp);
        fscanf(fp, "%d\n", &nz);
    
        fclose(fp);

        printf("Read_parameter is OK!\n");
    
        printf("nx, ny, nz is %d, %d, %d\n", nx, ny, nz);

        printf("The Number of total Processes is %d!\n", n_processe);
    }

    int tmp[4];

    if(my_id == 0){
        tmp[0] = nx;
        tmp[1] = ny;
        tmp[2] = nz;

        tmp[3] = N;
    }

    MPI_Bcast(tmp, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if(my_id != 0){
        nx = tmp[0];
        ny = tmp[1];
        nz = tmp[2];

        N = tmp[3];
    }

    NZ = nz/N;

    if(my_id < nz%N) NZ += 1;

    NPZ = (int*)malloc(N * sizeof(int));
    NP = (int*)malloc(N * sizeof(int));

    memset((void*)NPZ, 0, N);
    memset((void*)NP, 0, N);

    for(int i = 0; i < N; i++){
        if(i < nz%N){
            NPZ[i] = (int)nz/N + 1;
        }else{
            NPZ[i] = (int)nz/N;
        }
        NP[0] = 0;
        if(i != 0) NP[i] = NP[i-1] + NPZ[i-1];
    }

    yy = (double*)malloc(ny * _DOU);
    d1 = (double*)malloc(ny * _DOU);
    u1 = (double*)malloc(ny * _DOU);
    v1 = (double*)malloc(ny * _DOU);
    T1 = (double*)malloc(ny * _DOU);

    d2 = (double*)malloc(nx * ny * _DOU); 
    u2 = (double*)malloc(nx * ny * _DOU);
    v2 = (double*)malloc(nx * ny * _DOU);
    w2 = (double*)malloc(nx * ny * _DOU);
    T2 = (double*)malloc(nx * ny * _DOU);

    if(my_id == 0) printf("Read_parameter is OK!\n");
}

void Read_flow1d_inlet(){

    if(my_id == 0){
     
        if((fp = fopen("flow1d-inlet.dat", "r")) == NULL){
            printf("Can't open this file: 'flow1d-inlet.dat'\n");
            exit(0);
        }
     
        fgets(str, 100, fp);
        
        for(int j = 0; j < ny; j++){
     
            fscanf(fp, "%lf%lf%lf%lf%lf\n", &yy[j], &d1[j], &u1[j], &v1[j], &T1[j]);
     
        }
     
        fclose(fp);
    }

    MPI_Bcast(yy, ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(d1, ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u1, ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v1, ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(T1, ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(my_id == 0) printf("Read flow1d-inlet.dat is OK!\n");
}

void Creat_flow2d(){
    double (*pd2)[nx] = (double(*)[nx])(d2);
    double (*pu2)[nx] = (double(*)[nx])(u2);
    double (*pv2)[nx] = (double(*)[nx])(v2);
    double (*pw2)[nx] = (double(*)[nx])(w2);
    double (*pT2)[nx] = (double(*)[nx])(T2);

    for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
            pd2[j][i] = d1[j];
            pu2[j][i] = u1[j];
            pv2[j][i] = v1[j];
            pw2[j][i] = 0;
            pT2[j][i] = T1[j];
        }
    }
    if(my_id == 0) printf("Creat_flow2d is OK!\n");
}

#define FWRITE_MPI(ptr , size , num , stream){\
    int _ENCODE_START = num * size;\
    int _ENCODE_END = num * size;\
    MPI_File_write(stream, &_ENCODE_START, 1,  MPI_INT, &status);\
    MPI_File_write(stream, ptr, size,  MPI_DOUBLE, &status);\
    MPI_File_write(stream, &_ENCODE_END, 1,  MPI_INT, &status);\
    }

void Output_flow3d(){
    int nstep = 0;
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    double ntime = 0.;
    int DI = sizeof(int) + sizeof(double);

    MPI_File_open(MPI_COMM_WORLD, "flow3d0.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);
    MPI_File_write_all(tmp_file, &nstep, 1, MPI_INT, &status);
    MPI_File_write_all(tmp_file, &ntime, 1, MPI_DOUBLE, &status);
    MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);

    if(my_id == 0) printf("Write d ...\n");
    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, d2, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    if(my_id == 0) printf("Write u ...\n");
    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, u2, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    if(my_id == 0) printf("Write v ...\n");
    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, v2, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    if(my_id == 0) printf("Write w ...\n");
    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, w2, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    if(my_id == 0) printf("Write T ...\n");
    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, T2, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_close(&tmp_file);

    if(my_id == 0) printf("Wirte flow3d0.dat is OK!\n");
}

#undef FWRITE_MPI

void Finalize(){
    free(yy);
    free(d1);
    free(u1);
    free(v1);
    free(T1);

    free(d2);
    free(u2);
    free(v2);
    free(w2);
    free(T2);

    MPI_Finalize();
}

void output1d(int n, double *xx){
    for(int i = 0; i < n; i++){
        printf("%d\t%lf\n", i, xx[i]);
    }
}

void output2d(int nx, int ny, double *xx){
    double (*x)[nx] = (double(*)[nx])(xx);
    for(int j = 0; j < ny; j++){
        for(int i =0; i < nx; i++){
            printf("%d\t%d\t%lf\n", i, j, x[j][i]);
        }
    }
}
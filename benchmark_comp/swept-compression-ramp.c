//Mesh for swept-compression-ramp, coded by Dglin, 2019-11
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.1415926535897932

FILE *fp;
MPI_File tmp_file;
MPI_Status status;

char str[100];

int my_id, n_processe;
int Tec1, Tec2, Mesh2D;
int nx, ny, nz, N, NZ, *NPZ, *NP;
int nx_inlet, nx_conner, nx_buff;

double dx_inlet, Lx_inlet, Lx_conner, alfax_buff;
double sw_angle, comp_angle; 
double High, hy_wall, Lz;

double hx0, R;

double *xa, *xb, *ya, *yb, *xx, *yy, *xs, *ys, *x3d, *y3d, *z3d;

void mpi_init(int *Argc, char ***Argv);
void Read_parameter();
void gridx_inlet(int nx, double Length, double dx1, double dx2, double *xx);
void gridx();
void gridxy();
void gridxyz();
void Wall_Orthonormalization();
void output();
void Finalize();
void output1d(int n);
void output2d(int nx, int ny, double *xx);
void output3d(int nx, int ny, int nz, double *xx);

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();

    gridx_inlet(nx_inlet + 1, Lx_inlet, dx_inlet, hx0, xa);

    gridx();

    gridxy();

    Wall_Orthonormalization();

    gridxyz();

    output();

    //output1d(nx);

    //output2d(nx, ny, xx);

    Finalize();

    return 0;
}

void mpi_init(int *Argc , char *** Argv){

	MPI_Init(Argc, Argv);

    MPI_Comm_rank(MPI_COMM_WORLD , &my_id);
    
    MPI_Comm_size(MPI_COMM_WORLD, &n_processe);

}

void Read_parameter(){
    if(my_id == 0){
        if((fp = fopen("swept-compression-grid.in", "r")) == NULL){
            printf("Can't open this file: 'swept-compression-grid.in'\n");
            exit(0);
        }
    
        fgets(str, 100, fp);
        fscanf(fp, "%d%d%d%d\n", &nx,&ny,&nz,&N);
    
        fgets(str, 100, fp);
        fscanf(fp, "%d%d%d\n", &nx_inlet, &nx_conner, &nx_buff);
    
        fgets(str, 100, fp);
        fscanf(fp, "%lf%lf%lf%lf\n", &dx_inlet, &Lx_inlet, &Lx_conner, &alfax_buff);
    
        fgets(str, 100, fp);
        fscanf(fp, "%lf%lf\n", &sw_angle, &comp_angle);
    
        fgets(str, 100, fp);
        fscanf(fp, "%lf%lf%lf\n", &High, &hy_wall, &Lz);
        
        fgets(str, 100, fp);
        fscanf(fp, "%d%d%d\n", &Tec1, &Mesh2D, &Tec2);
    
        fclose(fp);
    
        if(nx_inlet + nx_conner + nx_buff != nx){
            printf("Error ! nx_conner+nx_inlet+nx_buff != nx\n");
            exit(0);
        }

        printf("Read_parameter is OK!\n");

        printf("The Number of total Processes is %d!\n", N);
    }

    int tmp1[10];
    double tmp2[9];
    
    if(my_id == 0){
        tmp1[0] = nx;
        tmp1[1] = ny;
        tmp1[2] = nz;

        tmp1[3] = N;

        tmp1[4] = nx_inlet;
        tmp1[5] = nx_conner;
        tmp1[6] = nx_buff;

        tmp1[7] = Tec1;
        tmp1[8] = Mesh2D;
        tmp1[9] = Tec2;

        tmp2[0] = dx_inlet;
        tmp2[1] = Lx_inlet;
        tmp2[2] = Lx_conner;
        tmp2[3] = alfax_buff;

        tmp2[4] = sw_angle;
        tmp2[5] = comp_angle;

        tmp2[6] = High;
        tmp2[7] = hy_wall;
        tmp2[8] = Lz;
    }

    MPI_Bcast(tmp1, 10, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmp2, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(my_id != 0){
        nx = tmp1[0];
        ny = tmp1[1];
        nz = tmp1[2];

        N = tmp1[3];

        nx_inlet = tmp1[4];
        nx_conner = tmp1[5];
        nx_buff = tmp1[6];

        Tec1 = tmp1[7];
        Mesh2D = tmp1[8];
        Tec2 = tmp1[9];

        dx_inlet = tmp2[0];
        Lx_inlet = tmp2[1];
        Lx_conner = tmp2[2];
        alfax_buff = tmp2[3];

        sw_angle = tmp2[4];
        comp_angle = tmp2[5];

        High = tmp2[6];
        hy_wall = tmp2[7];
        Lz = tmp2[8];
    }

    sw_angle = sw_angle * PI / 180.;
    comp_angle = comp_angle * PI / 180.;
    hx0 = 2. * Lx_conner/(nx_conner - 1);
    R = Lx_conner/tan(comp_angle/2.) - High;

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

    if(NP[N-1] != nz-NPZ[N-1]) printf("NP is wrong![debug]\n");

    xa = (double*)malloc(nx * sizeof(double));
    xb = (double*)malloc(nx * sizeof(double));

    ya = (double*)malloc(nx * sizeof(double));
    yb = (double*)malloc(nx * sizeof(double));

    xx = (double*)malloc(nx * ny * sizeof(double));
    yy = (double*)malloc(nx * ny * sizeof(double));

    if(N != n_processe){
        if(my_id == 0) printf("The Number of total Processes is not equal to N!\n");
        MPI_Finalize();
        exit(0);
    }
}

void gridx_inlet(int nx, double Length, double dx1, double dx2, double *xx){
    double dx, x, Ah1, Sh0, Sh1;

    dx = Length/(nx - 1.);

    if(my_id == 0) printf("dx1 is %lf, dx0 is %lf, dx2 is %lf!\n", dx1, dx, dx2);

    if((dx < dx1 && dx < dx2) || (dx > dx1 && dx > dx2)){
        if(my_id == 0) printf("warning !  dx0 should between dx1 and dx2 !!!\n");
    } 

    for(int i = 1; i <= nx; i++){
        x = (i -1.)/(nx - 1.);
        Ah1 = (1. - 2. * (x - 1.)) * x * x;
        Sh0 = x * pow(x - 1.,2);
        Sh1 = (x - 1.) * x * x;
        *(xx + i - 1) = Length * Ah1 + dx1 * (nx -1.) * Sh0 + dx2 * (nx - 1.) * Sh1;
    }
}

void gridx(){
    double seta, dx;
//inlet region
    for(int i = 0; i < nx_inlet; i++){
        xa[i] = xa[i] - Lx_conner - Lx_inlet;
        ya[i] = 0.;

        xb[i] = xa[i];
        yb[i] = High;
    }
//conner region
    for(int i = 0; i < nx_conner; i++){
        seta = comp_angle * i /(nx_conner - 1.);

        if(seta < comp_angle/2){
            xa[i + nx_inlet] = i * hx0 - Lx_conner;
            ya[i + nx_inlet] = 0.;
        }else
        {
            xa[i + nx_inlet] = (i * hx0 - Lx_conner) * cos(comp_angle);
            ya[i + nx_inlet] = (i * hx0 - Lx_conner) * sin(comp_angle);
        }
        
        xb[i + nx_inlet] = R * sin(seta) - Lx_conner;
        yb[i + nx_inlet] = R * (1. - cos(seta)) + High;
    }
//buff region
    for(int i = 0; i < nx_buff; i++){
        dx = (xa[i + nx_inlet + nx_conner -1] - xa[i + nx_inlet + nx_conner - 2]) * alfax_buff;

        xa[i + nx_inlet + nx_conner] = xa[i + nx_inlet + nx_conner -1] + dx;
        ya[i + nx_inlet + nx_conner] = ya[i + nx_inlet + nx_conner -1] + dx * tan(comp_angle);

        xb[i + nx_inlet + nx_conner] = xb[i + nx_inlet + nx_conner -1] + dx;
        yb[i + nx_inlet + nx_conner] = yb[i + nx_inlet + nx_conner -1] + dx * tan(comp_angle);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0) printf("Comput gridx is OK!\n");
}

void gridxy(){
    double SL, dy, delta, bnew, fb, fbx, s, sy, a, b = 3.5, continue_b = 0;

    double (*x)[nx] = (double(*)[nx])(xx);
    double (*y)[nx] = (double(*)[nx])(yy);

    for(int i = 0; i < nx; i++){
        SL = sqrt(pow(xb[i] - xa[i], 2) + pow(yb[i] - ya[i], 2));
        dy = 1. / (ny - 1.);
        delta = hy_wall / SL;
        
        do{ 
            continue_bnew:;
            fb = (exp(b / (ny - 1.)) - 1.)/(exp(b) - 1.) - delta;
            fbx = (exp(b / (ny - 1.))/(ny - 1.)*(exp(b) - 1.) - 
            (exp(b / (ny - 1.)) - 1.)*exp(b))/pow(exp(b) - 1., 2);
            bnew = b - fb / fbx;
            b = bnew;
            continue_b += 1;
        }
        while(fabs(bnew - b) > 1.e-6);
        if(continue_b < 10) goto continue_bnew; 
        
        a = 1./(exp(b) - 1.);

        for(int j = 0; j < ny; j++){
            s = j * dy;
            sy = a * (exp(s * b) - 1);

            x[j][i] = xa[i] + (xb[i] - xa[i]) * sy;
            y[j][i] = ya[i] + (yb[i] - ya[i]) * sy;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0) printf("Comput gridxy is OK!\n");
}

void gridxyz(){
    double z, tmp_inlet, tmp_conner;

    x3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    y3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    z3d = (double*)malloc(nx * ny * NZ * sizeof(double));

    double (*x)[nx] = (double (*)[nx])(xx);
    double (*y)[nx] = (double (*)[nx])(yy);

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);


        //for(int i = 0; i < nx_conner; i++){
//
        //    sx[j][i + nx_inlet] = sx[j][nx_inlet - 1] + x[j][i + nx_inlet] - tmp_inlet;
        //    sy[j][i + nx_inlet] = y[j][i + nx_inlet]; 
        //    x[j][i + nx_inlet] = x[j][nx_inlet - 1] + x[j][i + nx_inlet] - tmp_inlet;
        //}
//
        //for(int i = 0; i < nx_buff; i++){
//
        //    lx_buff = x[j][nx - 1] - tmp_conner;
        //    lx_buff_left = lx_buff + (N - my_id - 1) * Lz * tan(sw_angle)/N;
        //    lx_buff_right = lx_buff + (N - my_id) * Lz * tan(sw_angle)/N;
//
        //    sx[j][i + nx_inlet + nx_conner] = (x[j][i +nx_inlet + nx_conner] - 
        //    tmp_conner) * lx_buff_right / lx_buff + sx[j][nx_inlet + nx_conner - 1];
        //    
        //    sy[j][i + nx_inlet + nx_conner] = (y[j][i +nx_inlet + nx_conner] - 
        //    y[j][nx_inlet + nx_conner - 1]) * lx_buff_right / lx_buff + y[j][nx_inlet + nx_conner - 1];

        //    x[j][i + nx_inlet + nx_conner] = (x[j][i +nx_inlet + nx_conner] - 
        //    tmp_conner) * lx_buff_left / lx_buff + x[j][nx_inlet + nx_conner - 1];
        //}
//    }
    
    for(int k = 0; k < NZ; k++){

        z = k*Lz*(NPZ[my_id] - 1) / ((NZ - 1)*(nz - 1)) + NP[my_id]*Lz/(nz - 1);

        for(int j = 0; j < ny; j++){

            tmp_inlet = x[j][nx_inlet - 1];

            for(int i = 0; i < nx_inlet; i++){

                xx3d[k][j][i] = (x[j][i] - x[j][0])*(Lx_inlet - Lz*tan(sw_angle) + Lz/(nz - 1)*(k+NP[my_id])*tan(sw_angle))/Lx_inlet + x[j][0];
                yy3d[k][j][i] = y[j][i];
                zz3d[k][j][i] = z;
            }

            for(int i = nx_inlet; i < nx; i++){

                xx3d[k][j][i] = xx3d[k][j][nx_inlet - 1] + x[j][i] - tmp_inlet;;
                yy3d[k][j][i] = y[j][i];
                zz3d[k][j][i] = z;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0) printf("Comput gridxyz is OK!\n");
}

void Wall_Orthonormalization(){
    int Maxstep = 10, Jmax = 20;
    double b = 3., qn, qnx, qny, s;
    double *dx, *dy, *dx1, *dy1;

    double (*x)[nx] = (double(*)[nx])(xx);
    double (*y)[nx] = (double(*)[nx])(yy);

    dx = (double *)malloc(nx * sizeof(double));
    dy = (double *)malloc(nx * sizeof(double));
    dx1 = (double *)malloc(nx * sizeof(double));
    dy1 = (double *)malloc(nx * sizeof(double));

    for(int mstep = 0; mstep < Maxstep; mstep++){
        for(int j = 1; j < Jmax; j++){
            dx[0] = 0.;
            dy[0] = 0.;
            dx[nx - 1] = 0.;
            dy[nx - 1] = 0.;

            for(int i = 1; i < nx - 1; i++){
                qn = sqrt(pow(x[j][i + 1] - x[j][i - 1],2) + pow(y[j][i + 1] - y[j][i - 1], 2));
                qnx = -(y[j][i + 1] - y[j][i - 1])/qn;
                qny =  (x[j][i + 1] - x[j][i - 1])/qn;
                s = (x[j][i] - x[j - 1][i])*qnx + (y[j][i] - y[j - 1][i])*qny;

                dx[i] = s * qnx - (x[j][i] - x[j - 1][i]);
                dy[i] = s * qny - (y[j][i] - y[j - 1][i]);
            }

            for(int i = 1; i < nx - 1; i++){
                dx1[i] = (dx[i - 1] + 2. * dx[i] + dx[i + 1])/4.;
                dy1[i] = (dy[i - 1] + 2. * dy[i] + dy[i + 1])/4.;

                x[j][i] = x[j][i] + exp(-b * j / Jmax)/2. * dx1[i];
                y[j][i] = y[j][i] + exp(-b * j / Jmax)/2. * dy1[i];
            }

        }
    }
    free(dx);
    free(dy);
    free(dx1);
    free(dy1);

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0) printf("Wall_Orthonormalization is OK!\n");
}

#define FWRITE(ptr , size , num , stream){\
    int _ENCODE_START = num * size;\
    int _ENCODE_END = num * size;\
    fwrite(&_ENCODE_START, sizeof(int), 1, stream);\
    fwrite(ptr, size, num, stream);\
    fwrite(&_ENCODE_END, sizeof(int), 1, stream);\
    }

#define FWRITE_MPI(ptr , size , num , stream){\
    int _ENCODE_START = num * size;\
    int _ENCODE_END = num * size;\
    MPI_File_write(stream, &_ENCODE_START, 1,  MPI_INT, &status);\
    MPI_File_write(stream, ptr, size,  MPI_DOUBLE, &status);\
    MPI_File_write(stream, &_ENCODE_END, 1,  MPI_INT, &status);\
    }

void output(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    double (*x)[nx] = (double(*)[nx])(xx);
    double (*y)[nx] = (double(*)[nx])(yy);

    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    if(my_id == 0){
        fp = fopen("y1d.dat", "w");
        
        for(int j = 0; j < ny; j++){
           fprintf(fp, "%15.6f\n", y[j][0]);
        }
    
        fclose(fp);

        if(Tec1 == 1){
            fp = fopen("gridxy.dat", "w");
    
            fprintf(fp, "variables=x,y\n");
            fprintf(fp, "zone i=%d ,j=%d\n", nx, ny);
        
            for(int j = 0; j < ny; j++){
                for(int i = 0; i < nx; i++){
                    fprintf(fp, "%15.6f%15.6f\n", x[j][i], y[j][i]);
                }
            }
    
            fclose(fp);
        }

        if(Mesh2D == 1){
            fp = fopen("OCFD2d-Mesh.dat", "w");
            FWRITE(xx, sizeof(double), num, fp);
            FWRITE(yy, sizeof(double), num, fp);
            fclose(fp);
        }
    }
    
    MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    if(my_id == 0) printf("Write X3d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, x3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("Write Y3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, y3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("Write Z3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, z3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);

    if(Tec2 == 1){
        if(my_id == 0){
            fp = fopen("grid.dat", "w");
            fprintf(fp, "variables=x,y,z\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, nz);
        }
    
        for(int n = 0; n < N; n++){
            if(my_id == 0){
                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ny; j++){
                        for(int i = 0; i < nx; i++){
                            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[k][j][i], yy3d[k][j][i], zz3d[k][j][i]);
                        }
                    }
                }
            }
            if(my_id != 0){
                MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != N-1){
                MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
        }

        if(my_id == 0) fclose(fp);
    }
}

void Finalize(){
    free(xa);
    free(ya);

    free(xb);
    free(yb);

    free(xx);
    free(yy);

    free(x3d);
    free(y3d);
    free(z3d);

    MPI_Finalize();
}

void output1d(int n){
    fp = fopen("grid1d.dat", "w");
    for(int i = 0; i < n; i++){
        fprintf(fp, "%d\t%16.8f%16.8f%16.8f%16.8f\n", i, *(xa + i), *(ya + i), *(xb + i), *(yb + i));
    }
    fclose(fp);
}

void output2d(int nx, int ny, double *xx){
    double (*x)[nx] = (double(*)[nx])(xx);
    for(int j = 0; j < ny; j++){
        for(int i =0; i < nx; i++){
            printf("%d\t%d\t%lf\n", i, j, x[j][i]);
        }
    }
}

void output3d(int nx, int ny, int nz, double *xx){
    double (*x)[ny][nx] = (double(*)[ny][nx])(xx);
    for(int k = 0; k < nz; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                printf("%d\t%d\t%d\t%lf\n", i, j, k, x[k][j][i]);
            }
        }
    }
}

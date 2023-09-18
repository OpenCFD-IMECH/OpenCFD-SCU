//coded by Dgl
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "mpi.h"
#include "pthread.h"

#define PI 3.141592653589793

FILE *fp;
FILE *fp1;
MPI_Status status;

char  str[1000];
int init[3];
int my_id, n_processe;
int nx, ny, nz, NZ, *NPZ, *NP, *head;
double Re, Ama, Gamma, Pr, T_Ref, Tr, Tw, Amu, Cp, hh, tmp, Place;
double *x3d, *y3d, *z3d;
double *pd, *pu, *pv, *pw, *pT, *BC_rpara;

typedef struct configItem_
{
    char name[1000];
    char value[1000];
} configItem;


configItem configList[8] = {
    {"GRID_3D", 0},                 //0
    {"RE", 0},                      //1
    {"AMA", 0},                     //2
    {"GAMMA", 0},                   //3
    {"PR", 0},                      //4
    {"T_REF", 0},                   //5
    {"BC_RPARA", 0},                //6
    {"PLACE", 0},                   //7
};

void mpi_init(int *Argc, char ***Argv);
void Data_malloc();
void Read_parameter();
void Read_mesh();
void Read_data();
void Write_data_new();
void Write_data_new_extension();
void Read_mesh_1d();
void Read_data_1d();
void Write_grid_format();
void Write_data_format();
void Write_dataxy2d_first_format();
void Write_dataxy2d_first_format_total();
void Write_datayz2d_format();
void Write_dataxz2d_format();
void Write_datayz2d1_format();
void Write_dataxy2d1_format();
void Write_dataxyz3d_format();
void Write_flow1d_inlet();
void Write_cf2d_format();
void Write_cf3d_format();
void Write_cf3d_format_total();
void Write_dataxz2d_cf_double_cone(int i0);
void Write_dataxy2d_cf_compress_ramp(int i0);
//void Write_dataxy2d_yp();
void Finalize();

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    Read_parameter();
    Data_malloc();

    Read_mesh();
    Read_data();

    //Write_data_new();
    //Write_data_new_extension();

    //Write_cf2d_format();
    //Write_dataxy2d_first_format();
    //Write_grid_format();
    //Write_data_format();
    //Write_cf3d_format();
//    Write_datayz2d_format();
//    Write_dataxz2d_format();
    //Write_datayz2d1_format();
    //Write_dataxyz3d_format();

    //Write_flow1d_inlet();

//    Write_dataxy2d1_format();

    Write_dataxy2d_cf_compress_ramp(Place);
    //Write_dataxy2d_yp();

//-------double-cone---------------------
    {
        //Write_cf2d_format();
        //Write_dataxy2d_first_format_total();
        //Write_cf3d_format_total();
        //Write_datayz2d_format();
        //Write_dataxz2d_format();
        //Write_dataxz2d_cf_double_cone(Place);
    }


    Finalize();

    return 0;
}

void mpi_init(int *Argc , char *** Argv){

	MPI_Init(Argc, Argv);

    MPI_Comm_rank(MPI_COMM_WORLD , &my_id);
    
    MPI_Comm_size(MPI_COMM_WORLD, &n_processe);

}

int ExtarctItem(char *src, char *name, char *value){
    char *eq, *lf;
    eq = strchr(src, '=');
    lf = strchr(src, '\n');

    if(eq != NULL && lf != NULL){
        *lf = '\0';
        strncpy(name, src, eq-src);
        strcpy(value, eq+1);
        return 1;
    }

    return 0;
}


void ModifyItem(char *name, char *buff){
    while(*name != '\0')
    {
        if(*name != ' '){
            *buff = *name;
            buff++;
        }
        name++;
    }
}


void SearchItem(FILE *file, configItem *List, int configNum){
    int N = 1000;
    char buff[N];
    char name[N];
    char value[N];

    rewind(file);

    while(fgets(buff, N, file))
    {
        if(ExtarctItem(buff, name, value)){
            memset(buff, 0, strlen(buff));
            ModifyItem(name, buff);
            for(int i = 0; i < configNum; i++){
                if(strcmp(buff, List[i].name) == 0){
                    strcpy(List[i].value, value);
                }
            }
            memset(name, 0, strlen(name));
        }
    }
}


void RemovalNUM(char *buff){
    int i, j;

    for(i=j=0; buff[i]!='\0'; i++){
        if(buff[i]<'0' || buff[i]>'9')
            buff[j++] = buff[i];
    }

    buff[j] = '\0';
}


int StringToInteger(char *buff){
    int value = 0;

    while(*buff != '\0')
    {
        if(*buff>='0' && *buff<='9')
            value = value*10 + *buff - '0';

        buff++;
    }
    
    return value;
}


int ItemNUM(FILE *file, char *Item_name, int *NameNUM){
    int N = 1000;
    int i = 0;
    char buff[N];
    char name[N];
    char value[N];

    rewind(file);

    while(fgets(buff, N, file))
    {
        if(ExtarctItem(buff, name, value)){
            memset(buff, 0, strlen(buff));
            ModifyItem(name, buff);

            RemovalNUM(buff);

            if(strcmp(buff, Item_name) == 0){
                i += 1;
                *NameNUM = StringToInteger(name);
                NameNUM++;
            }

            memset(name, 0, strlen(name));
        }
    }

    return i;
}


int PartItem(char *src, char part[][1000]){
    const char blank[2] = " ";
    int num = 0;

    char *buff;

    buff = strtok(src, blank);

    while (buff != NULL)
    {
        strcpy(part[num], buff);
        buff = strtok(NULL, blank);

        num += 1;
    }

    return num;
}


void Read_parameter(){

    char Part_buff[50][1000];
    int configNum = sizeof(configList)/sizeof(configItem);

    if(my_id == 0){
        int nr;
        FILE *file = fopen("opencfd-scu.in","r");

        if(file == NULL){
            printf("\033[31mopencfd-scu.in is not find!\033[0m\n");
            exit(-1);
        }


        SearchItem(file, configList, configNum);

        sscanf(configList[0].value, "%d%d%d", &nx, &ny, &nz);
        sscanf(configList[1].value, "%lf", &Re);
        sscanf(configList[2].value, "%lf", &Ama);
        sscanf(configList[3].value, "%lf", &Gamma);
        sscanf(configList[4].value, "%lf", &Pr);

        sscanf(configList[5].value, "%lf", &T_Ref);

        printf("Computation start...\nRe is %lf\nAma is %lf\nGamma is %lf\nPr is %lf\nT_Ref is %lf\n",
            Re, Ama, Gamma, Pr, T_Ref);

        BC_rpara = (double*)malloc(sizeof(double)*100);
        nr = PartItem(configList[6].value, Part_buff);
        for(int i=0;i<nr;i++) sscanf(Part_buff[i],"%lf",BC_rpara+i);

        Tw = BC_rpara[0];
        Tr = 1 + 0.89 * (Gamma - 1.0) / 2.0 * Ama * Ama;

        printf("Tw is %lf\n", Tw);

        sscanf(configList[7].value, "%lf", &Place);

        printf("Place is %lf\n", Place);

        fclose(file);

    }
//-----------------------------------------------------------------------------------------------------
    int tmp1[3];
    double tmp2[7];

    if(my_id == 0){
        tmp1[0] = nx;
        tmp1[1] = ny;
        tmp1[2] = nz;

        tmp2[0] = Re;
        tmp2[1] = Ama;
        tmp2[2] = Gamma;
        tmp2[3] = Pr;

        tmp2[4] = T_Ref;
        tmp2[5] = Tw;
        tmp2[6] = Place;
    }

    MPI_Bcast(tmp1, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tmp2, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(my_id != 0){
        nx = tmp1[0];
        ny = tmp1[1];
        nz = tmp1[2];

        Re = tmp2[0];
        Ama = tmp2[1];
        Gamma = tmp2[2];
        Pr = tmp2[3];

        T_Ref = tmp2[4];
        Tw = tmp2[5];
        Place = tmp2[6];
    }
//-------------------------------------------------------------------------------------------

    Amu = 1.0/Re*(1.0 + 110.4/T_Ref)*sqrt(Tw*Tw*Tw)/(110.4/T_Ref + Tw);
    Cp = Gamma/((Gamma - 1)*Ama*Ama);

//-------------------------------------------------------------------------------------------

    NZ = nz/n_processe;

    if(my_id < nz%n_processe) NZ += 1;

    NPZ = (int*)malloc(n_processe * sizeof(int));
    NP = (int*)malloc(n_processe * sizeof(int));

    memset((void*)NPZ, 0, n_processe*sizeof(int));
    memset((void*)NP, 0, n_processe*sizeof(int));

    for(int i = 0; i < n_processe; i++){
        if(i < nz%n_processe){
            NPZ[i] = (int)nz/n_processe + 1;
        }else{
            NPZ[i] = (int)nz/n_processe;
        }
        NP[0] = 0;
        if(i != 0) NP[i] = NP[i-1] + NPZ[i-1];//偏移
    }

    if(NP[n_processe-1] != nz-NPZ[n_processe-1]) printf("NP is wrong![debug]\n");
}

#define Malloc_Judge(Memory)\
    {   if(Memory == NULL){\
        printf("Memory allocate error ! Can not allocate enough momory !!!\n");\
        exit(0); }\
    }

void Data_malloc(){
    x3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(x3d);

    y3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(y3d);

    z3d = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(z3d);  


    head = (int*)malloc(5 * sizeof(int));

    pd = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pd);

    pu = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pu);

    pv = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pv);

    pw = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pw); 

    pT = (double*)malloc(nx * ny * NZ * sizeof(double));
    Malloc_Judge(pT);
}

#undef Malloc_Judge

#define FREAD(ptr , size , num , stream) \
    {   int tmp_buffer;\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
        fread(ptr , size , num , stream);\
        fread(&tmp_buffer , sizeof(int) , 1 , stream);\
    }

void Read_mesh_1d(){
    if(my_id == 0){
        if((fp = fopen("OCFD3d-Mesh.dat", "r")) == NULL){
            printf("Can't open this file: 'OCFD3d-Mesh.dat'\n");
            exit(0);
        }

        int num = nx * ny;
        printf("Read OCFD3d-Mesh.dat-X ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(x3d + num * k, sizeof(double), num, fp);
        }

        printf("Read OCFD3d-Mesh.dat-Y ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(y3d + num * k, sizeof(double), num, fp);
        }

        printf("Read OCFD3d-Mesh.dat-Z ...\n\n");

        for(int k = 0; k < 2; k++){
            FREAD(z3d + num * k, sizeof(double), num, fp);
        }

        fclose(fp);
    }
}

void Read_data_1d(){
    if(my_id == 0){
        if((fp = fopen("opencfd.ana", "r")) == NULL){
            printf("Can't open this file: 'opencfd.ana'\n");
            exit(0);
        }

        int num = nx * ny;
        printf("Read opencfd.ana-d ...\n");

        fread(head , sizeof(int) , 5 , fp);

        for(int k = 0; k < 2; k++){
            FREAD(pd + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-u ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pu + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-v ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pv + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-w ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pw + num * k, sizeof(double), num, fp);
        }

        printf("Read opencfd.ana-T ...\n");

        for(int k = 0; k < 2; k++){
            FREAD(pT + num * k, sizeof(double), num, fp);
        }

        fclose(fp);
    }
}

#undef FREAD

void Write_flow1d_inlet(){
    double tmp[ny][4];

    if(my_id == 0){
            char str[100];
            double tmp1;
            fp = fopen("flow1d-inlet.dat", "r");
            printf("read inlet boundary data: flow-inlet.dat\n");
            fgets(str, 100, fp);
            for(int j = 0; j < ny; j++){
                fscanf(fp, "%lf%lf%lf%lf%lf\n", &tmp1, &tmp[j][0], &tmp[j][1], &tmp[j][2], &tmp[j][3]);
            }
            fclose(fp);

            fp = fopen("flow1d-inlet.dat", "w");
            fscanf(fp, "variables=y, d, u, v, T\n");
            for(int j = 0; j < ny-1; j++){
                fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, tmp[j][0], tmp[j][1], tmp[j][2], tmp[j][3]);
                fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, (tmp[j][0]+tmp[j+1][1])/2, (tmp[j][1]+tmp[j+1][1])/2, (tmp[j][2]+tmp[j+1][2])/2, (tmp[j][3]+tmp[j+1][3])/2);
            }

            fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, tmp[ny-1][0], tmp[ny-1][1], tmp[ny-1][2], tmp[ny-1][3]);
            fprintf(fp, "%lf%lf%lf%lf%lf\n", tmp1, tmp[ny-1][0], tmp[ny-1][1], tmp[ny-1][2], tmp[ny-1][3]);
	    fclose(fp);
    }
}

void Read_mesh(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);

    if(my_id == 0) printf("READ X3d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, x3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ Y3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, y3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ Z3d ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, z3d+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}


void Read_data(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "opencfd.ana", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);
	MPI_File_read_all(tmp_file, init, 1, MPI_INT, &status);
    MPI_File_read_all(tmp_file, init+1, 1, MPI_DOUBLE, &status);
	MPI_File_seek(tmp_file, sizeof(int), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pd+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ u ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pu+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ v ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pv+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ w ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pw+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("READ T ...\n");

    for(int k = 0; k < NZ; k++){
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_read(tmp_file, pT+num*k, num,  MPI_DOUBLE, &status);
        MPI_File_read(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}


void Write_data_new_extension(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    int DI = sizeof(int) + sizeof(double);
    MPI_File tmp_file;
    double *tmp2d;
    int nx_new = nx - 25;
    int num_new = nx_new*ny;

    tmp2d = (double*)malloc(nx*ny * sizeof(double));

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double (*tmp)[nx] = (double (*)[nx])(tmp2d);


    MPI_File_open(MPI_COMM_WORLD, "opencfd.md", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);
	MPI_File_write_all(tmp_file, &init, 1, MPI_INT, &status);
    MPI_File_write_all(tmp_file, &init[1], 1, MPI_DOUBLE, &status);
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);

    if(my_id == 0) printf("WRITE d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = d[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = d[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE u ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = u[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = u[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE v ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = v[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = v[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE w ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = w[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = w[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE T ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = T[k][j][i];
            }
            for(int i = nx_new; i < nx; i++){
                tmp[j][i] = T[k][j][nx_new-1];
            }
        }
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}


void Write_data_new(){
    int num = nx * ny;
    int num_byte = nx * ny * sizeof(double);
    int DI = sizeof(int) + sizeof(double);
    MPI_File tmp_file;
    double *tmp2d;
    int nx_new = nx - 10;
    int num_new = nx_new*ny;

    tmp2d = (double*)malloc(num_new * sizeof(double));

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double (*tmp)[nx_new] = (double (*)[nx_new])(tmp2d);

            fp = fopen("flow1d-inlet.dat", "w");
            fscanf(fp, "variables=y, d, u, v, T\n");
            for(int j = 0; j < ny; j++){
                fprintf(fp, "%15.6lf%15.6lf%15.6lf%15.6lf%15.6lf\n", 1.0, d[0][j][250], u[0][j][250], v[0][j][250], T[0][j][250]);
            }
	    fclose(fp);
exit(0);
    MPI_File_open(MPI_COMM_WORLD, "opencfd.md", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    MPI_File_seek(tmp_file, 0, MPI_SEEK_SET);
        
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);
	MPI_File_write_all(tmp_file, &init, 1, MPI_INT, &status);
    MPI_File_write_all(tmp_file, &init[1], 1, MPI_DOUBLE, &status);
	MPI_File_write_all(tmp_file, &DI, 1, MPI_INT, &status);

    if(my_id == 0) printf("WRITE d ...\n");

    MPI_File_seek(tmp_file, NP[my_id]*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);
    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = d[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE u ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = u[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE v ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = v[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE w ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = w[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }

    MPI_File_seek(tmp_file, (nz-NZ)*(num_new*sizeof(double) + 2*sizeof(int)), MPI_SEEK_CUR);

    if(my_id == 0) printf("WRITE T ...\n");

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx_new; i++){
                tmp[j][i] = T[k][j][i];
            }
        }
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
        MPI_File_write(tmp_file, tmp2d, num_new,  MPI_DOUBLE, &status);
        MPI_File_write(tmp_file, &num_byte, 1,  MPI_INT, &status);
    }
    //output3d(nx, ny, nz, z3d);
    MPI_File_close(&tmp_file);
}

void Write_grid_format(){
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    if(my_id == 0) printf("Write grid.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc(div * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    for(int m = 0; m < div; m++){
        if(my_id == 0){
            sprintf(filename, "grid%02d.dat", m);
            fp = fopen(filename, "w");

            fprintf(fp, "variables=x,y,z\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", MP[m], ny, nz);
            fclose(fp);
        }
    }

    for(int n = 0; n < n_processe; n++){
        for(int m = 0; m < div; m++){
            sprintf(filename, "grid%02d.dat", m);
    
            if(my_id == 0){
                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ny; j++){
                        for(int i = 0; i < MP[m]; i++){
                            int tmp = MP_offset[m] + i;
                            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[k][j][tmp], yy3d[k][j][tmp], zz3d[k][j][tmp]);
                        }
                    }
                }

                fclose(fp);
            }
        }

        if(my_id != 0){
            MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}


void Write_data_format(){
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    if(my_id == 0) printf("Write grid.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc(div * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    for(int m = 0; m < div; m++){
        if(my_id == 0){
            sprintf(filename, "grid%02d.dat", m);
            fp = fopen(filename, "w");

            fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", MP[m], ny, nz);
            fclose(fp);
        }
    }

    for(int n = 0; n < n_processe; n++){
        for(int m = 0; m < div; m++){
            sprintf(filename, "grid%02d.dat", m);
    
            if(my_id == 0){
                fp = fopen(filename, "a");

                for(int k = 0; k < NPZ[n]; k++){
                    for(int j = 0; j < ny; j++){
                        for(int i = 0; i < MP[m]; i++){
                            int tmp = MP_offset[m] + i;
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[k][j][tmp], yy3d[k][j][tmp], zz3d[k][j][tmp], 
                            d[k][j][tmp], u[k][j][tmp], v[k][j][tmp], w[k][j][tmp], T[k][j][tmp]);
                        }
                    }
                }

                fclose(fp);
            }
        }

        if(my_id != 0){
            MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);

            MPI_Send(pd, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);

            MPI_Recv(pd, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}


void Write_dataxy2d_first_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double cf, Tk;

    if(my_id == 0){
        printf("Write dataxy2d.dat\n");

        fp = fopen("dataxy.dat", "w");
        fprintf(fp, "variables=x,y,z,d,u,v,w,T,cf,Tk\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 2*ny-1, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][0][i] - T[0][0][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                yy3d[1][j][i], zz3d[1][j][i], d[1][j][i], u[1][j][i], v[1][j][i], w[1][j][i], T[1][j][i], cf, Tk);
            }
        }
        for(int j = ny-2; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][ny-1][i] - T[0][ny-1][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                -yy3d[1][j][i], zz3d[1][j][i], d[1][j][i], u[1][j][i], v[1][j][i], w[1][j][i], T[1][j][i], cf, Tk);
            }
        }

        fclose(fp);
    }
}


void Write_dataxy2d_first_format_total(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double cf, Tk;

    if(my_id == 0){
        printf("Write dataxy2d.dat\n");

        fp = fopen("dataxy.dat", "w");
        fprintf(fp, "variables=x,y,z,d,u,v,w,T,cf,Tk\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                hh = sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                          (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                          (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/hh;

                Tk = 2*Amu*Cp/Pr*(T[1][0][i] - T[0][0][i])/hh;

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                yy3d[1][j][i], zz3d[1][j][i], d[1][j][i], u[1][j][i], v[1][j][i], w[1][j][i], T[1][j][i], cf, Tk);
            }
        }

        fclose(fp);
    }
}


void Write_cf2d_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf, Tk;

    if(my_id == 0){
        printf("Write cf2d.dat\n");

        fp = fopen("cf2d_bottom.dat", "w");
        fprintf(fp, "variables=x,cf,Tk\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int i = 0; i < nx; i++){
            hh = sqrt((xx3d[1][0][i] - xx3d[0][0][i])*(xx3d[1][0][i] - xx3d[0][0][i]) + 
                      (yy3d[1][0][i] - yy3d[0][0][i])*(yy3d[1][0][i] - yy3d[0][0][i]) + 
                      (zz3d[1][0][i] - zz3d[0][0][i])*(zz3d[1][0][i] - zz3d[0][0][i]));

            cf = 2*Amu*sqrt(u[1][0][i]*u[1][0][i] + v[1][0][i]*v[1][0][i] + w[1][0][i]*w[1][0][i])/hh;

            Tk = 2*Amu*Cp/Pr*(T[1][0][i] - T[0][0][i])/hh;

            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[1][0][i], cf, Tk);
        }

        fclose(fp);

        fp = fopen("cf2d_top.dat", "w");
        fprintf(fp, "variables=x,cf,Tk\n");
        fprintf(fp, "zone i=%d\n", nx);

        for(int i = 0; i < nx; i++){
            hh = sqrt((xx3d[1][ny-1][i] - xx3d[0][ny-1][i])*(xx3d[1][ny-1][i] - xx3d[0][ny-1][i]) + 
                      (yy3d[1][ny-1][i] - yy3d[0][ny-1][i])*(yy3d[1][ny-1][i] - yy3d[0][ny-1][i]) + 
                      (zz3d[1][ny-1][i] - zz3d[0][ny-1][i])*(zz3d[1][ny-1][i] - zz3d[0][ny-1][i]));

            cf = 2*Amu*sqrt(u[1][ny-1][i]*u[1][ny-1][i] + v[1][ny-1][i]*v[1][ny-1][i] + w[1][ny-1][i]*w[1][ny-1][i])/hh;

            Tk = 2*Amu*Cp/Pr*(T[1][ny-1][i] - T[0][ny-1][i])/hh;

            fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[1][ny-1][i], cf, Tk);
        }

        fclose(fp);
    }
}

void Write_cf3d_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf;

    if(my_id == 0){
        printf("Write cf3d.dat\n");

        fp = fopen("cf3d.dat", "w");
        fprintf(fp, "variables=x,y,z,cf\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 2*ny-1, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/
                           sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                                (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                                (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                yy3d[1][j][i], zz3d[1][j][i], cf);
            }
        }
        for(int j = ny-2; j >= 0; j--){
            for(int i = 0; i < nx; i++){
                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/
                           sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                                (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                                (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));
                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                -yy3d[1][j][i], zz3d[1][j][i], cf);
            }
        }

        fclose(fp);
    }
}


void Write_cf3d_format_total(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf;

    if(my_id == 0){
        printf("Write cf3d.dat\n");

        fp = fopen("cf3d.dat", "w");
        fprintf(fp, "variables=x,y,z,cf\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                cf = 2*Amu*sqrt(u[1][j][i]*u[1][j][i] + v[1][j][i]*v[1][j][i] + w[1][j][i]*w[1][j][i])/
                           sqrt((xx3d[1][j][i] - xx3d[0][j][i])*(xx3d[1][j][i] - xx3d[0][j][i]) + 
                                (yy3d[1][j][i] - yy3d[0][j][i])*(yy3d[1][j][i] - yy3d[0][j][i]) + 
                                (zz3d[1][j][i] - zz3d[0][j][i])*(zz3d[1][j][i] - zz3d[0][j][i]));

                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[1][j][i],
                yy3d[1][j][i], zz3d[1][j][i], cf);
            }
        }

        fclose(fp);
    }
}

void Write_datayz2d1_format(){//写出一个yz截面的数据，7/10位置
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    pd_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pu_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pv_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pw_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pT_buff  = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny] = (double(*)[ny])x3d_buff;
    double (*yy3d_buff)[ny] = (double(*)[ny])y3d_buff;
    double (*zz3d_buff)[ny] = (double(*)[ny])z3d_buff;
    double (*ppd_buff)[ny]  = (double(*)[ny])pd_buff;
    double (*ppu_buff)[ny]  = (double(*)[ny])pu_buff;
    double (*ppv_buff)[ny]  = (double(*)[ny])pv_buff;
    double (*ppw_buff)[ny]  = (double(*)[ny])pw_buff;
    double (*ppT_buff)[ny]  = (double(*)[ny])pT_buff;


    if(my_id == 0) printf("Write datayz2d_1.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc((div+1) * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    MP_offset[div] = nx;

    int m=7;
    int tmp = MP_offset[m+1];

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            xx3d_buff[k][j] = xx3d[k][j][tmp];
            yy3d_buff[k][j] = yy3d[k][j][tmp];
            zz3d_buff[k][j] = zz3d[k][j][tmp];
            ppd_buff[k][j]  = d[k][j][tmp];
            ppu_buff[k][j]  = u[k][j][tmp];
            ppv_buff[k][j]  = v[k][j][tmp];
            ppw_buff[k][j]  = w[k][j][tmp];
            ppT_buff[k][j]  = T[k][j][tmp];
        }
    }

    if(my_id == 0){
        sprintf(filename, "datayz%02d.dat", m);
        fp = fopen(filename, "w");

        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, 2*ny-1, nz);
        fclose(fp);
    }


    for(int n = 0; n < n_processe; n++){

        sprintf(filename, "datayz%02d.dat", m);

        if(my_id == 0){
            fp = fopen(filename, "a");
    
            for(int k = 0; k < NPZ[n]; k++){

                for(int j = 0; j < ny; j++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
                        yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
                        ppw_buff[k][j], ppT_buff[k][j]);
                }
                for(int j = ny-2; j >= 0; j--){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
                        -yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
                        ppw_buff[k][j], ppT_buff[k][j]);
                }
            }

            fclose(fp);
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}

void Write_datayz2d_format(){
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(ny * NZ * sizeof(double));
    pd_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pu_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pv_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pw_buff  = (double*)malloc(ny * NZ * sizeof(double));
    pT_buff  = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny] = (double(*)[ny])x3d_buff;
    double (*yy3d_buff)[ny] = (double(*)[ny])y3d_buff;
    double (*zz3d_buff)[ny] = (double(*)[ny])z3d_buff;
    double (*ppd_buff)[ny]  = (double(*)[ny])pd_buff;
    double (*ppu_buff)[ny]  = (double(*)[ny])pu_buff;
    double (*ppv_buff)[ny]  = (double(*)[ny])pv_buff;
    double (*ppw_buff)[ny]  = (double(*)[ny])pw_buff;
    double (*ppT_buff)[ny]  = (double(*)[ny])pT_buff;


    if(my_id == 0) printf("Write datayz2d.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc((div+1) * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    MP_offset[div] = nx;

    for(int m = 0; m < div; m++){
        if(my_id == 0){
            sprintf(filename, "datayz%02d.dat", m);
            fp = fopen(filename, "w");

            fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, ny, nz);
//            fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", 1, 2*ny-1, nz);
            fclose(fp);
        }
    }


    for(int m = 0; m < div; m++){
        int tmp = MP_offset[m+1];

        for(int k = 0; k < NZ; k++){
            for(int j = 0; j < ny; j++){
                xx3d_buff[k][j] = xx3d[k][j][tmp];
                yy3d_buff[k][j] = yy3d[k][j][tmp];
                zz3d_buff[k][j] = zz3d[k][j][tmp];
                ppd_buff[k][j]  = d[k][j][tmp];
                ppu_buff[k][j]  = u[k][j][tmp];
                ppv_buff[k][j]  = v[k][j][tmp];
                ppw_buff[k][j]  = w[k][j][tmp];
                ppT_buff[k][j]  = T[k][j][tmp];
            }
        }
        for(int n = 0; n < n_processe; n++){

            sprintf(filename, "datayz%02d.dat", m);

            if(my_id == 0){
                fp = fopen(filename, "a");
    
                for(int k = 0; k < NPZ[n]; k++){

                    for(int j = 0; j < ny; j++){
                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
                            yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
                            ppw_buff[k][j], ppT_buff[k][j]);
                    }
//                    for(int j = ny-2; j >= 0; j--){
//                            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j],
//                            -yy3d_buff[k][j], zz3d_buff[k][j], ppd_buff[k][j], ppu_buff[k][j], ppv_buff[k][j], 
//                            ppw_buff[k][j], ppT_buff[k][j]);
//                    }
                }

                fclose(fp);
            }


            if(my_id != 0){
                MPI_Send(x3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(y3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(z3d_buff, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pd_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pu_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pv_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pw_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
                MPI_Send(pT_buff , ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            }

            if(my_id != n_processe-1){
                MPI_Recv(x3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(y3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(z3d_buff, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pd_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pu_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pv_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pw_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(pT_buff , ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            }
        }
    }
}


void Write_dataxz2d_format(){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    x3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    y3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    z3d_buff = (double*)malloc(nx * NZ * sizeof(double));
    pd_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pu_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pv_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pw_buff  = (double*)malloc(nx * NZ * sizeof(double));
    pT_buff  = (double*)malloc(nx * NZ * sizeof(double));

    double (*xx3d_buff)[nx] = (double(*)[nx])x3d_buff;
    double (*yy3d_buff)[nx] = (double(*)[nx])y3d_buff;
    double (*zz3d_buff)[nx] = (double(*)[nx])z3d_buff;
    double (*ppd_buff)[nx]  = (double(*)[nx])pd_buff;
    double (*ppu_buff)[nx]  = (double(*)[nx])pu_buff;
    double (*ppv_buff)[nx]  = (double(*)[nx])pv_buff;
    double (*ppw_buff)[nx]  = (double(*)[nx])pw_buff;
    double (*ppT_buff)[nx]  = (double(*)[nx])pT_buff;


    if(my_id == 0) printf("Write dataxz2d.dat\n");
    char filename[100];

    for(int k = 0; k < NZ; k++){
        for(int i = 0; i < nx; i++){
            xx3d_buff[k][i] = xx3d[k][9][i];
            yy3d_buff[k][i] = yy3d[k][9][i];
            zz3d_buff[k][i] = zz3d[k][9][i];
            ppd_buff[k][i]  = d[k][9][i];
            ppu_buff[k][i]  = u[k][9][i];
            ppv_buff[k][i]  = v[k][9][i];
            ppw_buff[k][i]  = w[k][9][i];
            ppT_buff[k][i]  = T[k][9][i];
        }
    }

    if(my_id == 0){
        sprintf(filename, "dataxz.dat");
        fp = fopen(filename, "w");

        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, 1, nz);
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int i = 0; i < nx; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][i],
                        yy3d_buff[k][i], zz3d_buff[k][i], ppd_buff[k][i], ppu_buff[k][i], ppv_buff[k][i], 
                        ppw_buff[k][i], ppT_buff[k][i]);
                }

            }
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    if(my_id == 0) fclose(fp);
}


void Write_dataxy2d1_format(){
    int div = 10;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);


    if(my_id == 0) printf("Write dataxy2d_1.dat\n");
    char filename[100];

    int m=(nz-1)/2, n, id = n_processe-1;

    for(int i = 0; i < n_processe-1; i++){
        if(NP[i] < m && NP[i+1] > m){
            id = i;
        }
    }

    n = m - NP[id];

    if(my_id == id){
        sprintf(filename, "dataxy%06d.dat", m);
        fp = fopen(filename, "w");

        fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, 1);

    
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < nx; i++){
                    fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d[n][j][i],
                    yy3d[n][j][i], zz3d[n][j][i], d[n][j][i], u[n][j][i], v[n][j][i], w[n][j][i], 
                    T[n][j][i]);
            }
        }

        fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


void average_data_xy(double (*data2d)[nx], double (*data3d)[ny][nx]){
    double *pdata_buff = (double*)malloc(nx*ny*sizeof(double));
    double (*data_buff)[nx] = (double (*)[nx])(pdata_buff);

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            data2d[j][i] = 0.0;
            data_buff[j][i] = 0.0;
            for(int k=0; k<NZ; k++){
                data_buff[j][i] += data3d[k][j][i];
            }
        }
    }

    MPI_Reduce(&data_buff[0][0], &data2d[0][0], nx*ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            data2d[j][i] /= nz;
        }
    }
}


void average_data_xz(double (*data2d)[nx], double (*data3d)[ny][nx]){
    double *pdata_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*data_buff)[nx] = (double (*)[nx])(pdata_buff);

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            data_buff[k][i] = 0.0;
            for(int j=0; j<ny; j++){
                data_buff[k][i] += data3d[k][j][i];
            }
            data_buff[k][i] /= ny;
        }
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for (int i=0; i<nx; i++)
                {
                    data2d[NP[n] + k][i] = data_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            MPI_Send(pdata_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(pdata_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}


void comput_yh_us_compress_ramp(double (*yh2d)[nx], double (*us2d)[nx], double (*u2d)[nx], double (*v2d)[nx], int i0){//针对压缩折角，计算点到壁面距离与平行壁面速度(展向总和)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double length, length_min;
    double seta0 = 24.0*PI/180.0;
    //double seta0 = 0;
    int start_point = 8000;
    double seta;

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            us2d[j][i] = 0.0;
//            if(i <= i0){
//                length_min = yy3d[0][j][i];
//            }else{
                length_min = 10000;
                
                for(int iw=0; iw<nx; iw++){
                    length = sqrt(pow(xx3d[0][j][iw] - xx3d[0][0][iw], 2) + 
                    pow(yy3d[0][j][iw] - yy3d[0][0][iw], 2));

                    if(length <= length_min) length_min = length;//计算格点与壁面最近的距离
                }
//            }
            yh2d[j][i] = length_min;

            if(xx3d[0][0][i] <= 0){
                seta = 0.0;
            }else{
                seta = seta0;/* 如果是在角区，则将速度投影至避面平行方向*/
            }

            us2d[j][i] = u2d[j][i]*cos(seta) + v2d[j][i]*sin(seta);
        }
    }
}


void comput_zh_us_compress_ramp(double (*zh2d)[nx], double (*us2d)[nx]){//针对压缩折角，计算点到壁面距离与平行壁面速度(展向总和)
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);

    double length;
    double seta0 = 7.0*PI/180.0;
    double seta1 = 0.0;
    int start_point = 8000;
    double seta;

    double *pxx_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*xx_buff)[nx] = (double (*)[nx])(pxx_buff);

    double *pzz_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*zz_buff)[nx] = (double (*)[nx])(pzz_buff);

    double *puu_buff = (double*)malloc(nx*NZ*sizeof(double));
    double (*uu_buff)[nx] = (double (*)[nx])(puu_buff);

    double (*xx2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*zz2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    double *pVx = (double*)malloc(nx * ny * NZ * sizeof(double));

    double *pVseta = (double*)malloc(nx * ny * NZ * sizeof(double));

    double *pVr = (double*)malloc(nx * ny * NZ * sizeof(double));


    double (*Vx)[ny][nx] = (double (*)[ny][nx])(pVx);
    double (*Vseta)[ny][nx] = (double (*)[ny][nx])(pVseta);
    double (*Vr)[ny][nx] = (double (*)[ny][nx])(pVr);

    for(int i=0; i<nx; i++){
        for(int k=0; k<NZ; k++){
            for(int j=0; j<ny; j++){
                Vr[k][j][i] = v[k][j][i]*sin(2*PI/(ny-1)*j) + w[k][j][i]*cos(2*PI/(ny-1)*j);
                Vseta[k][j][i] = v[k][j][i]*cos(2*PI/(ny-1)*j) - w[k][j][i]*sin(2*PI/(ny-1)*j);

                Vx[k][j][i] = u[k][j][i]*cos(seta0) + Vr[k][j][i]*sin(seta0);
                Vr[k][j][i] = -u[k][j][i]*sin(seta0) + Vr[k][j][i]*cos(seta0);
            }
        }
    }

    for(int k=0; k<NZ; k++){
        for(int i=0; i<nx; i++){
            xx_buff[k][i] = xx3d[k][0][i];
            zz_buff[k][i] = zz3d[k][0][i];

            uu_buff[k][i] = 0.0;

            for (int j=0; j<ny; j++)
            {
                uu_buff[k][i] += Vx[k][j][i];
            }

            uu_buff[k][i] /= ny;
        }
    }

    for(int n = 0; n < n_processe; n++){

        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for (int i=0; i<nx; i++)
                {
                    xx2d[NP[n] + k][i] = xx_buff[k][i];
                    zz2d[NP[n] + k][i] = zz_buff[k][i];
                    us2d[NP[n] + k][i] = uu_buff[k][i];
                }
            }
        }


        if(my_id != 0){
            MPI_Send(pxx_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pzz_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(puu_buff, nx*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(pxx_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pzz_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(puu_buff, nx*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    for(int k=0; k<nz; k++){
        for(int i=0; i<nx; i++){

            length = sqrt(pow(xx2d[k][i] - xx2d[0][i], 2) + 
            pow(zz2d[k][i] - zz2d[0][i], 2));

            zh2d[k][i] = length;
        }
    }
}

void Write_OCFDYZ_Mesh(int i0){
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double *x2d = (double*)malloc(ny * NZ * sizeof(double));
    double *y2d = (double*)malloc(ny * NZ * sizeof(double));
    double *z2d = (double*)malloc(ny * NZ * sizeof(double));

    double (*xx2d)[ny] = (double (*)[ny])(x2d);
    double (*yy2d)[ny] = (double (*)[ny])(y2d);
    double (*zz2d)[ny] = (double (*)[ny])(z2d);

    if(my_id == 0) 
    {
        fp = fopen("zz1d.dat", "w");
        for(int i = 0; i < nz; i++){
            fprintf(fp, "%d%15.6f\n", i, zz3d[i][0][i0]);
        }
        fclose(fp);
    }

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            xx2d[k][j] = xx3d[k][j][i0];
            yy2d[k][j] = yy3d[k][j][i0];
            zz2d[k][j] = zz3d[k][j][i0];
        }
    }

    if(my_id == 0){
        printf("Write OCFDYZ-Mesh.dat\n");
        fp = fopen("OCFDYZ-Mesh.dat", "w");
    }

    for(int n = 0; n < n_processe; n++){
        if(my_id == 0){
            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    fprintf(fp, "%15.6f%15.6f%15.6f\n", xx2d[k][j], yy2d[k][j], zz2d[k][j]);
                }
            }
        }

        if(my_id != 0){
            MPI_Send(x2d, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y2d, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z2d, ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x2d, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y2d, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z2d, ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }

    if(my_id == 0) fclose(fp);
}

void Write_dataxz2d_cf_double_cone(int i0){//针对顿锥问题的后处理，写出表面摩阻
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf, Tk, us1, us2, h1, h2, uy, P, xp, yp, zp;
    double *cf0 = (double*)malloc(nx*sizeof(double));
    double *Tk0 = (double*)malloc(nx*sizeof(double));

    double *Ut = (double*)malloc(nx*sizeof(double));
    double *Ret = (double*)malloc(nx*sizeof(double));
    double *up = (double*)malloc(nz*sizeof(double));
    double *uvd = (double*)malloc(nz*sizeof(double));

    double (*zh2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));//格点距离壁面距离
    double (*us2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));//延壁面水平方向投影过的流向速度

    double (*d2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));
    double (*cf2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*Tk2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*T2d)[nx] = (double(*)[nx])malloc(nx*nz*sizeof(double));

    average_data_xz(T2d, T);
    average_data_xz(d2d, d);

    comput_zh_us_compress_ramp(zh2d, us2d);

    Write_OCFDYZ_Mesh(i0);

    if(my_id == 0){

        for(int i = 0; i < nx; i++){

            us1 = us2d[1][i];
            us2 = us2d[2][i];

            h1 = zh2d[1][i];
            h2 = zh2d[2][i];

            uy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);//展向速度梯度之和

            cf0[i] = 2*Amu*uy;

            Tk0[i] = 2*Amu*Cp/Pr*(T2d[1][i] - T2d[0][i])/h1;//计算展向热流和
        }

        printf("Write cf2d.dat\n");

        fp = fopen("cf2d.dat", "w");
        fprintf(fp, "variables=x,cf,Tk,pw\n");
        fprintf(fp, "zone i=%d\n", nx);
        for(int i = 0; i < nx; i++){
            P = d2d[1][i]*T2d[1][i];
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], cf0[i], Tk0[i], P);
        }
        fclose(fp);

//--------------------------------------------------------------------------------------
        for(int i = 0; i < nx; i++){
            Ut[i] = sqrt( fabs(cf0[i]) / (2*d2d[0][i]) );//摩擦速度
            Ret[i] = d2d[0][i]*Ut[i]/Amu;//粘性尺度倒数
        }

        fp = fopen("xyzp.dat", "w");
        fprintf(fp, "zone i=%d\n", nx-1);
        for(int i = 1; i < nx; i++){
            xp = (xx3d[0][0][i] - xx3d[0][0][i-1])*Ret[i];
            yp = (yy3d[0][1][i] - yy3d[0][0][i])*Ret[i];
            zp = (zz3d[1][0][i] - zz3d[0][0][i])*Ret[i];

            fprintf(fp, "%15.6f%15.6f%15.6f\n", xp, yp, zp);
        }
        fclose(fp);


        printf("Write one-dimension profiles\n");

        double zp;
        printf("i0 is %d, Axx is %lf\n", i0, xx3d[0][0][i0]);

        up[0] = 0; 
        uvd[0] = 0;

        fp = fopen("U1d.dat", "w");
        fprintf(fp, "variables=yp,up,uvd,u_log\n");
        fprintf(fp, "zone i=%d\n", nz-2);
        for(int k = 1; k < nz-1; k++){
            zp = zh2d[k][i0]*Ret[i0];
            up[k] = us2d[k][i0]/Ut[i0];
            uvd[k] = uvd[k-1] + sqrt(d2d[k][i0]/d2d[0][i0])*(up[k] - up[k-1]);

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", zp, up[k], uvd[k], 2.44*log(zp)+5.1);
        }
        fclose(fp);

//---------------------------------------------------------------------------------------

        printf("Write delt\n");
        double delt0, delt1, delt2;
        int z0;

        fp = fopen("delta.dat", "w");

        for(int i = 0; i < nx; i++){
            delt1 = 0;
            delt2 = 0;
            for(int k = 0; k < nz; k++){
                if(us2d[k][i] > 0.99){
                    z0 = k-1;
                    goto end_comput_delt;
                }
            }

            end_comput_delt:;

            delt0 = zh2d[z0][i];//速度边界层厚度

            for(int k = 1; k <= z0; k++){
                delt1 += (1 - d2d[k][i]*us2d[k][i]/(d2d[z0][i]*us2d[z0][i]))*(zh2d[k][i] - zh2d[k-1][i]);
                delt2 += d2d[k][i]*us2d[k][i]/(d2d[z0][i]*us2d[z0][i])*(1 - us2d[k][i]/us2d[z0][i])*(zh2d[k][i] - zh2d[k-1][i]); 
            }
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], delt0, delt1, delt2);
        }
        fclose(fp);
    }
}


void Write_dataxy2d_cf_compress_ramp(int i0){//针对压缩折角问题的后处理，写出表面摩阻
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
    double cf, Tk, us1, us2, h1, h2, uy, P;
    double *cf0 = (double*)malloc(nx*sizeof(double));
    double *Tk0 = (double*)malloc(nx*sizeof(double));

    double *Ut = (double*)malloc(nx*sizeof(double));
    double *Ret = (double*)malloc(nx*sizeof(double));
    double *up = (double*)malloc(ny*sizeof(double));
    double *uvd = (double*)malloc(ny*sizeof(double));

    double (*yh2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));//格点距离壁面距离
    double (*us2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));//延壁面水平方向投影过的流向速度

    double (*d2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*u2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*v2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double (*T2d)[nx] = (double(*)[nx])malloc(nx*ny*sizeof(double));
    double xp, yp, zp;

    average_data_xy(u2d, u);
    average_data_xy(v2d, v);
    average_data_xy(T2d, T);
    average_data_xy(d2d, d);


    if(my_id == 0){
        
        comput_yh_us_compress_ramp(yh2d, us2d, u2d, v2d, i0);

        for(int i = 0; i < nx; i++){

            us1 = us2d[1][i];
            us2 = us2d[2][i];

            h1 = yh2d[1][i];
            h2 = yh2d[2][i];

            uy = (h2*h2*us1 - h1*h1*us2)/(h2*h2*h1 - h1*h1*h2);//展向速度梯度之和

            cf0[i] = 2*Amu*uy;

            Tk0[i] = Amu*(T2d[1][i] - T2d[0][i])/(h1*Pr*(Tr-Tw));//计算展向热流和
        }


        printf("Write cf2d.dat\n");

        fp = fopen("cf2d.dat", "w");
        fprintf(fp, "variables=x,cf,Tk,P\n");
        fprintf(fp, "zone i=%d\n", nx);
        for(int i = 0; i < nx; i++){
            P = d2d[1][i]*T2d[1][i];
       //     fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[0][0][i], cf0[i], Tk0[i]);
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], cf0[i], Tk0[i], P);
        }
        fclose(fp);

//--------------------------------------------------------------------------------------
        for(int i = 0; i < nx; i++){
            Ut[i] = sqrt( fabs(cf0[i]) / (2*d2d[0][i]) );//摩擦速度
            Ret[i] = d2d[0][i]*Ut[i]/Amu;//粘性尺度倒数
        }

        fp = fopen("xyzp.dat", "w");
        fprintf(fp, "zone i=%d\n", nx-1);
        for(int i = 1; i < nx; i++){
            xp = (xx3d[0][0][i] - xx3d[0][0][i-1])*Ret[i];
            yp = (yy3d[0][1][i] - yy3d[0][0][i])*Ret[i];
            zp = (zz3d[1][0][i] - zz3d[0][0][i])*Ret[i];

            fprintf(fp, "%15.6f%15.6f%15.6f\n", xp, yp, zp);
        }
        fclose(fp);

        printf("Write one-dimension profiles\n");

        double yp;
        printf("i0 is %d, Axx is %lf\n", i0, xx3d[0][0][i0]);

        up[0] = 0; 
        uvd[0] = 0;

        fp = fopen("U1d.dat", "w");
        fprintf(fp, "variables=yp,up,uvd,u_log\n");
        fprintf(fp, "zone i=%d\n", ny-2);
        for(int j = 1; j < ny-1; j++){
            yp = yh2d[j][i0]*Ret[i0];
            up[j] = us2d[j][i0]/Ut[i0];
            uvd[j] = uvd[j-1] + sqrt(d2d[j][i0]/d2d[0][i0])*(up[j] - up[j-1]);

            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", yp, up[j], uvd[j], 2.44*log(yp)+5.1);
        }
        fclose(fp);

//---------------------------------------------------------------------------------------

        printf("Write delt\n");
        double delt0, delt1, delt2;
        int j0;

        fp = fopen("delta.dat", "w");

        for(int i = 0; i < nx; i++){
            delt1 = 0;
            delt2 = 0;
            for(int j = 0; j < ny; j++){
                if(us2d[j][i] > 0.99){
                    j0 = j-1;
                    goto end_comput_delt;
                }
            }

            end_comput_delt:;

            delt0 = yh2d[j0][i];//速度边界层厚度

            for(int j = 1; j <= j0; j++){
                delt1 += (1 - d2d[j][i]*us2d[j][i]/(d2d[j0][i]*us2d[j0][i]))*(yh2d[j][i] - yh2d[j-1][i]);
                delt2 += d2d[j][i]*us2d[j][i]/(d2d[j0][i]*us2d[j0][i])*(1 - us2d[j][i]/us2d[j0][i])*(yh2d[j][i] - yh2d[j-1][i]); 
            }
            fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", xx3d[0][0][i], delt0, delt1, delt2);
        }
        fclose(fp);
    }
}

void Write_dataxyz3d_format(){
    int div = 10;
    int m = 7;
    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);

    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);

    double *x3d_buff, *y3d_buff, *z3d_buff, *pd_buff, *pu_buff, *pv_buff, *pw_buff, *pT_buff;

    if(my_id == 0) printf("Write dataxyz3d.dat\n");
    char filename[100];
    int *MP = (int*)malloc(div * sizeof(int));
    int *MP_offset = (int*)malloc((div+1) * sizeof(int));

    for(int i = 0; i < div; i++){
        if(i < nx%div){
            MP[i] = (int)nx/div + 1;
        }else{
            MP[i] = (int)nx/div;
        }

        MP_offset[0] = 0;

        if(i != 0) MP_offset[i] = MP_offset[i-1] + MP[i-1];
    }

    MP_offset[div] = nx;

    x3d_buff = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    y3d_buff = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    z3d_buff = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pd_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pu_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pv_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pw_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));
    pT_buff  = (double*)malloc(MP[m] * ny * NZ * sizeof(double));

    double (*xx3d_buff)[ny][MP[m]] = (double(*)[ny][MP[m]])x3d_buff;
    double (*yy3d_buff)[ny][MP[m]] = (double(*)[ny][MP[m]])y3d_buff;
    double (*zz3d_buff)[ny][MP[m]] = (double(*)[ny][MP[m]])z3d_buff;
    double (*ppd_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pd_buff;
    double (*ppu_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pu_buff;
    double (*ppv_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pv_buff;
    double (*ppw_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pw_buff;
    double (*ppT_buff)[ny][MP[m]]  = (double(*)[ny][MP[m]])pT_buff;

    int tmp;

    for(int k = 0; k < NZ; k++){
        for(int j = 0; j < ny; j++){
            for(int i = 0; i < MP[m]; i++){
                tmp = MP_offset[m] + i;
                xx3d_buff[k][j][i] = xx3d[k][j][tmp];
                yy3d_buff[k][j][i] = yy3d[k][j][tmp];
                zz3d_buff[k][j][i] = zz3d[k][j][tmp];
                ppd_buff[k][j][i]  = d[k][j][tmp];
                ppu_buff[k][j][i]  = u[k][j][tmp];
                ppv_buff[k][j][i]  = v[k][j][tmp];
                ppw_buff[k][j][i]  = w[k][j][tmp];
                ppT_buff[k][j][i]  = T[k][j][tmp];
            }
        }
    }

    if(my_id == 0){
        sprintf(filename, "data07.dat");
        fp = fopen(filename, "w");
        fclose(fp);
    }

    for(int n = 0; n < n_processe; n++){

        sprintf(filename, "data07.dat");

        if(my_id == 0){
            fp = fopen(filename, "a");
            if(n == 0){
                fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
                fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", MP[m], 2*ny-1, nz);
            }
    
            for(int k = 0; k < NPZ[n]; k++){
                for(int j = 0; j < ny; j++){
                    for(int i = 0; i < MP[m]; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppd_buff[k][j][i], ppu_buff[k][j][i], ppv_buff[k][j][i], 
                            ppw_buff[k][j][i], ppT_buff[k][j][i]);
                    }
                }
                for(int j = ny-2; j >= 0; j--){
                    for(int i = 0; i < MP[m]; i++){
                        fprintf(fp, "%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f%15.6f\n", xx3d_buff[k][j][i],
                            -yy3d_buff[k][j][i], zz3d_buff[k][j][i], ppd_buff[k][j][i], ppu_buff[k][j][i], ppv_buff[k][j][i], 
                            ppw_buff[k][j][i], ppT_buff[k][j][i]);
                    }

                }
            }

            fclose(fp);
        }


        if(my_id != 0){
            MPI_Send(x3d_buff, MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(y3d_buff, MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(z3d_buff, MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pd_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pu_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pv_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pw_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
            MPI_Send(pT_buff , MP[m]*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
        }

        if(my_id != n_processe-1){
            MPI_Recv(x3d_buff, MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(y3d_buff, MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(z3d_buff, MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pd_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pu_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pv_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pw_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(pT_buff , MP[m]*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
        }
    }
}



//void Write_grid(){
//    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
//    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
//    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);
//
//    printf("Write grid.dat\n");
//    if(my_id == 0){
//        fp = fopen("grid.dat", "w");
//        fprintf(fp, "variables=x,y,z\n");
//        fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, nz);
//    }
//
//    for(int n = 0; n < n_processe; n++){
//        
//        if(my_id == 0){
//            for(int k = 0; k < NPZ[n]; k++){
//                for(int j = 0; j < ny; j++){
//                    for(int i = 0; i < nx; i++){
//                        fprintf(fp, "%15.6f%15.6f%15.6f\n", xx3d[k][j][i], yy3d[k][j][i], zz3d[k][j][i]);
//                    }
//                }
//            }
//        }
//
//        if(my_id != 0){
//            MPI_Send(x3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
//            MPI_Send(y3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
//            MPI_Send(z3d, nx*ny*NPZ[my_id], MPI_DOUBLE, my_id-1, 1, MPI_COMM_WORLD);
//        }
//
//        if(my_id != n_processe-1){
//            MPI_Recv(x3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
//            MPI_Recv(y3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
//            MPI_Recv(z3d, nx*ny*NPZ[my_id+1], MPI_DOUBLE, my_id+1, 1, MPI_COMM_WORLD, &status);
//        }
//    }
//
//    if(my_id == 0) fclose(fp);
//}

//void Write_data(){
//    double (*xx3d)[ny][nx] = (double (*)[ny][nx])(x3d);
//    double (*yy3d)[ny][nx] = (double (*)[ny][nx])(y3d);
//    double (*zz3d)[ny][nx] = (double (*)[ny][nx])(z3d);
//
//    double (*d)[ny][nx] = (double (*)[ny][nx])(pd);
//    double (*u)[ny][nx] = (double (*)[ny][nx])(pu);
//    double (*v)[ny][nx] = (double (*)[ny][nx])(pv);
//    double (*w)[ny][nx] = (double (*)[ny][nx])(pw);
//    double (*T)[ny][nx] = (double (*)[ny][nx])(pT);
//
//    fp = fopen("opencfd.format", "w");
//
//    fprintf(fp, "variables=x,y,z,d,u,v,w,T\n");
//    fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, nz);
//    for(int k = 0; k < nz; k++){
//        for(int j = 0; j < ny; j++){
//            for(int i = 0; i < nx; i++){
//                fprintf(fp, "%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf%32.10lf\n", 
//                xx3d[k][j][i], yy3d[k][j][i], zz3d[k][j][i], 
//                d[k][j][i], u[k][j][i], v[k][j][i],
//                w[k][j][i], T[k][j][i]);
//            }
//        }
//    }
//
//    fclose(fp);
//}


void Finalize(){
    free(x3d);
    free(y3d);
    free(z3d);

    free(head);
    free(pd);
    free(pu);
    free(pv);
    free(pw);
    free(pT);
}

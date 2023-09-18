#include "test.h"
#include "parameters.h"

#include "stdio.h"
void write_block_me(char * name , REAL * u , int nx , int ny , int nz)    //[nx][ny][nz][n]
{
    REAL (*U)[nz][ny][nx] = (REAL(*)[nz][ny][nx])u;
    FILE * file = fopen(name , "w");
for(int n=0;n<5;n++){
    for(int k=0;k<nz;k++){
        for(int j=0;j<ny;j++){
            for(int i=0;i<nx;i++){
                fprintf(file , "i=%3d,j=%3d,k=%2d,n=%1d\t%32.10lf\n",i+1,j+1,k+1,n+1,U[n][k][j][i]);
            }
        }
    }
}
    fclose(file);
}

void write_block_me1(char * name , REAL * u , int nx , int ny , int nz)   //[nx][ny][nz]
{
    REAL (*U)[ny+2*LAP][nx+2*LAP] = (REAL(*)[ny+2*LAP][nx+2*LAP])u;
    FILE * file = fopen(name , "w");
    for(int k=0;k<nz+2*LAP;k++){
        for(int j=0;j<ny+2*LAP;j++){
            for(int i=0;i<nx+2*LAP;i++){
                fprintf(file , "i=%3d,j=%3d,k=%2d\t%32.10lf\n",i-LAP+1,j-LAP+1,k-LAP+1,U[k][j][i]);
            }
        }
    }
    fclose(file);
}

void write_block_me2(char * name , REAL * u , int nx , int ny , int nz)    //[nx+LAP][ny+LAP][nz+LAP][n]
{
    REAL (*U)[nz+2*LAP][ny+2*LAP][nx+2*LAP] = (REAL(*)[nz+2*LAP][ny+2*LAP][nx+2*LAP])u;
    FILE * file = fopen(name , "w");
for(int n=0;n<5;n++){
    for(int k=0;k<nz+2*LAP;k++){
        for(int j=0;j<ny+2*LAP;j++){
            for(int i=0;i<nx+2*LAP;i++){
                fprintf(file , "i=%3d,j=%3d,k=%2d,n=%1d\t%32.10lf\n",i-LAP+1,j-LAP+1,k-LAP+1,n+1,U[n][k][j][i]);
            }
        }
    }
}
    fclose(file);
}

void write_block_me3(char * name , REAL * u , int nx , int ny , int nz)    //[nx][ny][nz]
{
    REAL (*U)[ny][nx] = (REAL(*)[ny][nx])u;
    FILE * file = fopen(name , "w");
    for(int k=0;k<nz;k++){
        for(int j=0;j<ny;j++){
            for(int i=0;i<nx;i++){
                fprintf(file , "i=%3d,j=%3d,k=%2d\t%32.10lf\n",i+1,j+1,k+1,U[k][j][i]);
            }
        }
    }
    fclose(file);
}


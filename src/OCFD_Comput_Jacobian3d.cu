/*--------- This code runs only at the initial times -----------------------
读入计算网格 (Axx, Ayy, Azz),  计算Jocaiban 系数；  
该程序仅在初始化阶段运行 
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include "parameters.h"
#include "utility.h"
#include "OCFD_Comput_Jacobian3d.h"
#include "OCFD_Schemes_Choose.h"
#include "OCFD_mpi.h"
#include "OCFD_IO.h"

#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "OCFD_mpi_dev.h"
#include "commen_kernel.h"
#include "math.h"
#include "OCFD_ana.h"


#ifdef __cplusplus
extern "C"{
#endif


void Init_Jacobian3d()
{
    //    init with unit

    cuda_mem_value_init_warp(1.0 , pAjac_d->ptr , pAjac_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAxx_d->ptr , pAxx_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAyy_d->ptr , pAyy_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAzz_d->ptr , pAzz_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAkx_d->ptr , pAkx_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAky_d->ptr , pAky_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAkz_d->ptr , pAkz_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAix_d->ptr , pAix_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAiy_d->ptr , pAiy_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAiz_d->ptr , pAiz_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAsx_d->ptr , pAsx_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAsy_d->ptr , pAsy_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    cuda_mem_value_init_warp(1.0 , pAsz_d->ptr , pAsz_d->pitch , nx_2lap , ny_2lap , nz_2lap);
    {
        REAL * tmp;
        int tmp_size = (nx+2*LAP)*(ny+2*LAP)*(nz+2*LAP);
        tmp = pAxx ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAyy ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAzz ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAkx ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAky ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAkz ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAix ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAiy ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAiz ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAsx ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAsy ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAsz ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
        tmp = pAjac ;for(int i=0;i<tmp_size;i++) (*tmp++) = 1.0;
    }
    // -------------------------------------------------------------------------
    char filename1[100];
    MPI_File tmp_file;
    sprintf(filename1, "OCFD3d-Jacobi.dat");

    if(Init_stat == 0){
        int i,j,k;
        int klap , jlap,ilap;
        int i_off , j_off , k_off;
        int i_real, j_real, k_real;
        REAL r , d_r;
        REAL theta , d_theta , theta_0;

        REAL r0 = 1.0;
        REAL dr = 1.0;
        d_theta = PI / NY_GLOBAL;
        theta_0 = -PI*0.5;
        d_r = dr / NZ_GLOBAL;

        i_off = i_offset[npx];
        j_off = j_offset[npy];
        k_off = k_offset[npz];
        for(k = 0;k<nz;k++){
            klap = k+LAP;
            k_real = k + k_off;
            r = r0 + d_r * k_real;
            for(j=0;j<ny;j++){
                jlap = j+LAP;
                j_real = j + j_off;
                theta = theta_0 + d_theta * j_real;
                for(i=0;i<nx;i++){
                    ilap = i+LAP;
                    i_real = i + i_off;
                    *(pAxx + ilap + nx_2lap*jlap + nx_2lap*ny_2lap*klap) = i_real * hx;

                    *(pAyy + ilap + nx_2lap*jlap + nx_2lap*ny_2lap*klap) = r * cos(theta);
                    *(pAzz + ilap + nx_2lap*jlap + nx_2lap*ny_2lap*klap) = r * sin(theta);
                }
            }
        }
        if(npy == NPY0 - 1){
            jlap = ny - 1 + LAP;
            for(k = 0; k<nz ; k++){
                klap = k+LAP;
                k_real = k + k_off;
                r = r0 + d_r * k_real;
                for(i=0;i<nx;i++){
                    ilap = i+LAP;
                    *(pAyy + ilap + nx_2lap*jlap + nx_2lap*ny_2lap*klap) = 0.0;
                    *(pAzz + ilap + nx_2lap*jlap + nx_2lap*ny_2lap*klap) = r;
                }
            }
        }
    }else if(access(filename1, F_OK) == -1){ 
        if(my_id == 0) printf("read 3D mesh data: OCFD3d-Mesh.dat ...\n");
        
        MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Mesh.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);
        
        MPI_Offset offset = 0;

        read_3d1(tmp_file, offset, pAxx);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAyy);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAzz);
    
        MPI_File_close(&tmp_file);

        exchange_boundary_xyz(pAxx);
        exchange_boundary_xyz(pAyy);
        exchange_boundary_xyz(pAzz);
        
        memcpy_All(pAxx , pAxx_d->ptr , pAxx_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAyy , pAyy_d->ptr , pAyy_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAzz , pAzz_d->ptr , pAzz_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);

        Comput_Jacobian3d();
    }else{
        //The file not exist
        if(my_id == 0) printf("OCFD3d-Jacobi.dat is exit\nread 3D Jacobi data ...... ");
        MPI_File tmp_file;
        
        MPI_File_open(MPI_COMM_WORLD, "OCFD3d-Jacobi.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &tmp_file);
        
        MPI_Offset offset = 0;

        read_3d1(tmp_file, offset, pAxx);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAyy);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAzz);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAkx);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAky);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAkz);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAix);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAiy);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAiz);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAsx);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAsy);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAsz);
        offset += (2*sizeof(int) + NX_GLOBAL * NY_GLOBAL * sizeof(REAL)) * NZ_GLOBAL;
        read_3d1(tmp_file, offset, pAjac);
        
    
        MPI_File_close(&tmp_file);

        exchange_boundary_xyz(pAxx);
        exchange_boundary_xyz(pAyy);
        exchange_boundary_xyz(pAzz);
        exchange_boundary_xyz(pAkx);
        exchange_boundary_xyz(pAky);
        exchange_boundary_xyz(pAkz);
        exchange_boundary_xyz(pAix);
        exchange_boundary_xyz(pAiy);
        exchange_boundary_xyz(pAiz);
        exchange_boundary_xyz(pAsx);
        exchange_boundary_xyz(pAsy);
        exchange_boundary_xyz(pAsz);
        exchange_boundary_xyz(pAjac);
    
        memcpy_All(pAxx , pAxx_d->ptr , pAxx_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAyy , pAyy_d->ptr , pAyy_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAzz , pAzz_d->ptr , pAzz_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);

        memcpy_All(pAkx , pAkx_d->ptr , pAkx_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAky , pAky_d->ptr , pAky_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAkz , pAkz_d->ptr , pAkz_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAix , pAix_d->ptr , pAix_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAiy , pAiy_d->ptr , pAiy_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAiz , pAiz_d->ptr , pAiz_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAsx , pAsx_d->ptr , pAsx_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAsy , pAsy_d->ptr , pAsy_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAsz , pAsz_d->ptr , pAsz_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);
        memcpy_All(pAjac , pAjac_d->ptr , pAjac_d->pitch , H2D , nx_2lap , ny_2lap , nz_2lap);

    }

    ana_Jac();
}


void Comput_Jacobian3d(){

    boundary_Jac3d_Axx(); //only using the  boudary condition for Axx, Ayy, Azz

    if (my_id == 0)
        printf("Comput Jacobian 3D data ...\n");
    
    comput_Jac3d();
    if (my_id == 0)
        printf("Comput Jacobian 3D data OK\n");

    // ---------------
    exchange_boundary_xyz_packed_dev(pAkx_d);
    exchange_boundary_xyz_packed_dev(pAky_d);
    exchange_boundary_xyz_packed_dev(pAkz_d);
    exchange_boundary_xyz_packed_dev(pAix_d);
    exchange_boundary_xyz_packed_dev(pAiy_d);
    exchange_boundary_xyz_packed_dev(pAiz_d);
    exchange_boundary_xyz_packed_dev(pAsx_d);
    exchange_boundary_xyz_packed_dev(pAsy_d);
    exchange_boundary_xyz_packed_dev(pAsz_d);
    exchange_boundary_xyz_packed_dev(pAjac_d);

    boundary_Jac3d_Liftbody_Ajac(); //boudary condition for Axx, Ayy, Azz, Aix, Aiy, Aiz , ......

}

// ----------------------------------------------------------------------------

__global__ void comput_Jac3d_kernal(
    cudaField xi,
    cudaField xj,
    cudaField xk,
    cudaField yi,
    cudaField yj,
    cudaField yk,
    cudaField zi,
    cudaField zj,
    cudaField zk,
    cudaField Akx,
    cudaField Aky,
    cudaField Akz,
    cudaField Aix,
    cudaField Aiy,
    cudaField Aiz,
    cudaField Asx,
    cudaField Asy,
    cudaField Asz,
    cudaField Ajac,
    cudaJobPackage job
){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL xi1, xj1, xk1, yi1, yj1, yk1, zi1, zj1, zk1, Jac1;
        xi1 = get_Field(xi, x, y, z);
        xj1 = get_Field(xj, x, y, z);
        xk1 = get_Field(xk, x, y, z);
        yi1 = get_Field(yi, x, y, z);
        yj1 = get_Field(yj, x, y, z);
        yk1 = get_Field(yk, x, y, z);
        zi1 = get_Field(zi, x, y, z);
        zj1 = get_Field(zj, x, y, z);
        zk1 = get_Field(zk, x, y, z);
        Jac1 = 1.0 / (xi1 * yj1 * zk1 + yi1 * zj1 * xk1 + zi1 * xj1 * yk1 - zi1 * yj1 * xk1 - yi1 * xj1 * zk1 - xi1 * zj1 * yk1); //1./Jocabian = d(x,y,z)/d(i,j,k)
        get_Field_LAP(Ajac , x+LAP , y+LAP , z+LAP) = Jac1;
        get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP) = (yj1 * zk1 - zj1 * yk1);
        get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP) = (zj1 * xk1 - xj1 * zk1);
        get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP) = (xj1 * yk1 - yj1 * xk1);
        get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP) = (yk1 * zi1 - zk1 * yi1);
        get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP) = (zk1 * xi1 - xk1 * zi1);
        get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP) = (xk1 * yi1 - yk1 * xi1);
        get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP) = (yi1 * zj1 - zi1 * yj1);
        get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP) = (zi1 * xj1 - xi1 * zj1);
        get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP) = (xi1 * yj1 - yi1 * xj1);


        if(x == 0){
            for(int i = 0; i < LAP; i++){
                get_Field_LAP(Ajac, i, y+LAP, z+LAP) = get_Field_LAP(Ajac, x+LAP , y+LAP , z+LAP);
                get_Field_LAP(Akx, i, y+LAP, z+LAP) =  get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aky, i, y+LAP, z+LAP) =  get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Akz, i, y+LAP, z+LAP) =  get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aix, i, y+LAP, z+LAP) =  get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiy, i, y+LAP, z+LAP) =  get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiz, i, y+LAP, z+LAP) =  get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asx, i, y+LAP, z+LAP) =  get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asy, i, y+LAP, z+LAP) =  get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asz, i, y+LAP, z+LAP) =  get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);
            }
        }

        if(x == job.end.x-1){
            for(int i = 1; i <= LAP; i++){
                get_Field_LAP(Ajac, x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Ajac, x+LAP , y+LAP , z+LAP);
                get_Field_LAP(Akx,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aky,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Akz,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aix,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiy,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiz,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asx,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asy,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asz,  x+LAP+i, y+LAP, z+LAP) = get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);
            }
        }

        if(y == 0){
            for(int i = 0; i < LAP; i++){
                get_Field_LAP(Ajac, x+LAP, i, z+LAP) = get_Field_LAP(Ajac, x+LAP , y+LAP , z+LAP);
                get_Field_LAP(Akx, x+LAP, i, z+LAP) =  get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aky, x+LAP, i, z+LAP) =  get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Akz, x+LAP, i, z+LAP) =  get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aix, x+LAP, i, z+LAP) =  get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiy, x+LAP, i, z+LAP) =  get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiz, x+LAP, i, z+LAP) =  get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asx, x+LAP, i, z+LAP) =  get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asy, x+LAP, i, z+LAP) =  get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asz, x+LAP, i, z+LAP) =  get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);
            }
        }


        if(y == job.end.y-1){
            for(int i = 1; i <= LAP; i++){
                get_Field_LAP(Ajac, x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Ajac, x+LAP , y+LAP , z+LAP);
                get_Field_LAP(Akx,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aky,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Akz,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aix,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiy,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiz,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asx,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asy,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asz,  x+LAP, y+LAP+i, z+LAP) = get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);
            }
        }

        if(z == 0){
            for(int i = 0; i < LAP; i++){
                get_Field_LAP(Ajac, x+LAP, y+LAP, i) = get_Field_LAP(Ajac, x+LAP , y+LAP , z+LAP);
                get_Field_LAP(Akx,  x+LAP, y+LAP, i) =  get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aky,  x+LAP, y+LAP, i) =  get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Akz,  x+LAP, y+LAP, i) =  get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aix,  x+LAP, y+LAP, i) =  get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiy,  x+LAP, y+LAP, i) =  get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiz,  x+LAP, y+LAP, i) =  get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asx,  x+LAP, y+LAP, i) =  get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asy,  x+LAP, y+LAP, i) =  get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asz,  x+LAP, y+LAP, i) =  get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);
            }
        }

        if(z == job.end.z-1){
            for(int i = 1; i <= LAP; i++){
                get_Field_LAP(Ajac, x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Ajac, x+LAP , y+LAP , z+LAP);
                get_Field_LAP(Akx,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aky,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Akz,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aix,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiy,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Aiz,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asx,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asy,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);
                get_Field_LAP(Asz,  x+LAP, y+LAP, z+LAP+i) = get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);
            }
        }
	}
}

void comput_Jac3d()
{
    cudaField xi; xi.ptr = puk_d->ptr; xi.pitch = puk_d->pitch;
    cudaField xj; xj.ptr = pui_d->ptr; xj.pitch = pui_d->pitch;
    cudaField xk; xk.ptr = pus_d->ptr; xk.pitch = pus_d->pitch;
    cudaField yi; yi.ptr = pvk_d->ptr; yi.pitch = pvk_d->pitch;
    cudaField yj; yj.ptr = pvi_d->ptr; yj.pitch = pvi_d->pitch;
    cudaField yk; yk.ptr = pvs_d->ptr; yk.pitch = pvs_d->pitch;
    cudaField zi; zi.ptr = pwk_d->ptr; zi.pitch = pwk_d->pitch;
    cudaField zj; zj.ptr = pwi_d->ptr; zj.pitch = pwi_d->pitch;
    cudaField zk; zk.ptr = pws_d->ptr; zk.pitch = pws_d->pitch;

	cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap, nz_lap) );
    
    OCFD_dx0_jac(*pAxx_d, xi, job, BlockDim_X, &Stream[0], Jacbound[0]);
    OCFD_dx0_jac(*pAyy_d, yi, job, BlockDim_X, &Stream[0], Jacbound[0]);
    OCFD_dx0_jac(*pAzz_d, zi, job, BlockDim_X, &Stream[0], Jacbound[0]);
    OCFD_dy0_jac(*pAxx_d, xj, job, BlockDim_Y, &Stream[0], Jacbound[1]);
    OCFD_dy0_jac(*pAyy_d, yj, job, BlockDim_Y, &Stream[0], Jacbound[1]);
    OCFD_dy0_jac(*pAzz_d, zj, job, BlockDim_Y, &Stream[0], Jacbound[1]);
    OCFD_dz0_jac(*pAxx_d, xk, job, BlockDim_Z, &Stream[0], Jacbound[2]);
    OCFD_dz0_jac(*pAyy_d, yk, job, BlockDim_Z, &Stream[0], Jacbound[2]);
    OCFD_dz0_jac(*pAzz_d, zk, job, BlockDim_Z, &Stream[0], Jacbound[2]);

    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX/8 , BlockDimY , BlockDimZ , nx,ny,nz);
    job.setup( dim3(0,0,0) , dim3(nx,ny,nz) );

    CUDA_LAUNCH(( comput_Jac3d_kernal<<<griddim , blockdim>>>(xi,xj,xk,yi,yj,yk,zi,zj,zk,*pAkx_d,
            *pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,*pAjac_d,job) ));

}

// ------------------------------------------------------------------------
// Symmetry bounary at j=1 & j=ny_global

__global__ void boundary_Jac3d_kernal_y_r(cudaField pA, REAL value, cudaJobPackage job){
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y*blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z*blockIdx.z + threadIdx.z + job.start.z;

    if( x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(pA, x, y, z) =  value*get_Field_LAP(pA, x, 2*(ny_lap_d-1) - y, z);
    }
}

__global__ void boundary_Jac3d_kernal_y_l(cudaField pA, REAL value, cudaJobPackage job){
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y*blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z*blockIdx.z + threadIdx.z + job.start.z;

    if( x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(pA, x, y, z) =  value*get_Field_LAP(pA, x, 2*LAP - y, z);
    }
}


__global__ void boundary_Jac3d_kernal_y_ramp_wall_kernel(
    cudaField xx, 
    cudaField yy, 
    cudaField Akx,
    cudaField Aky,
    cudaField Akz,
    cudaField Aix,
    cudaField Aiy,
    cudaField Aiz,
    cudaField Asx,
    cudaField Asy,
    cudaField Asz,
    cudaField Ajac,
    REAL seta,
    cudaJobPackage job){
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y*blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z*blockIdx.z + threadIdx.z + job.start.z;

    if( x < job.end.x && y < job.end.y && z < job.end.z){
        if( get_Field_LAP(xx, x, LAP, z) <= 0.0){
            get_Field_LAP(Ajac, x, y, z) = get_Field_LAP(Ajac, x, 2*LAP-y, z);
            get_Field_LAP(Akx, x, y, z) =  get_Field_LAP(Akx, x, 2*LAP-y, z);
            get_Field_LAP(Aky, x, y, z) = -get_Field_LAP(Aky, x, 2*LAP-y, z);
            get_Field_LAP(Akz, x, y, z) =  get_Field_LAP(Akz, x, 2*LAP-y, z);
            get_Field_LAP(Aix, x, y, z) = -get_Field_LAP(Aix, x, 2*LAP-y, z);
            get_Field_LAP(Aiy, x, y, z) =  get_Field_LAP(Aiy, x, 2*LAP-y, z);
            get_Field_LAP(Aiz, x, y, z) = -get_Field_LAP(Aiz, x, 2*LAP-y, z);
            get_Field_LAP(Asx, x, y, z) =  get_Field_LAP(Asx, x, 2*LAP-y, z);
            get_Field_LAP(Asy, x, y, z) = -get_Field_LAP(Asy, x, 2*LAP-y, z);
            get_Field_LAP(Asz, x, y, z) =  get_Field_LAP(Asz, x, 2*LAP-y, z);

        }else{

            REAL dx = get_Field_LAP(xx, x, 2*LAP-y, z) - get_Field_LAP(xx, x, 2*LAP-y-1, z);
            REAL dy = get_Field_LAP(yy, x, 2*LAP-y, z) - get_Field_LAP(yy, x, 2*LAP-y-1, z);

            REAL tmpxx = fabs(-cos(2*seta) + sin(2*seta)*dy/dx);
            REAL tmpyy = fabs(cos(2*seta) + dx/dy*sin(2*seta));

            get_Field_LAP(Ajac, x, y, z) = tmpxx*tmpyy*get_Field_LAP(Ajac, x, 2*LAP-y, z);
            get_Field_LAP(Akx, x, y, z) =  tmpyy*get_Field_LAP(Akx, x, 2*LAP-y, z);
            get_Field_LAP(Aky, x, y, z) = -tmpxx*get_Field_LAP(Aky, x, 2*LAP-y, z);
            get_Field_LAP(Akz, x, y, z) =  tmpxx*tmpyy*get_Field_LAP(Akz, x, 2*LAP-y, z);
            get_Field_LAP(Aix, x, y, z) = -tmpyy*get_Field_LAP(Aix, x, 2*LAP-y, z);
            get_Field_LAP(Aiy, x, y, z) =  tmpxx*get_Field_LAP(Aiy, x, 2*LAP-y, z);
            get_Field_LAP(Aiz, x, y, z) = -tmpxx*tmpyy*get_Field_LAP(Aiz, x, 2*LAP-y, z);
            get_Field_LAP(Asx, x, y, z) =  tmpyy*get_Field_LAP(Asx, x, 2*LAP-y, z);
            get_Field_LAP(Asy, x, y, z) = -tmpxx*get_Field_LAP(Asy, x, 2*LAP-y, z);
            get_Field_LAP(Asz, x, y, z) =  tmpxx*tmpyy*get_Field_LAP(Asz, x, 2*LAP-y, z);

        }
    }
}

void boundary_Jac3d_kernal_y_ramp_wall(REAL seta){
    if (npy == 0)
    {
        seta = seta/PI;
        dim3 griddim , blockdim;
        cudaJobPackage job( dim3(LAP , 0 , LAP) , dim3(nx_lap , LAP , nz_lap) );

        CUDA_LAUNCH(( boundary_Jac3d_kernal_y_ramp_wall_kernel<<<griddim , blockdim>>>(*pAxx_d,*pAyy_d,*pAkx_d,
            *pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,*pAjac_d,seta,job) ));
    }
}

__global__ void boundary_Jac3d_kernal_z_cone_wall_kernel(
    cudaField xx, 
    cudaField zz, 
    cudaField Akx,
    cudaField Aky,
    cudaField Akz,
    cudaField Aix,
    cudaField Aiy,
    cudaField Aiz,
    cudaField Asx,
    cudaField Asy,
    cudaField Asz,
    cudaField Ajac,
    REAL seta1,
    REAL seta2,
    cudaJobPackage job){
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x + job.start.x;
    unsigned int y = blockDim.y*blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z*blockIdx.z + threadIdx.z + job.start.z;

    if( x < job.end.x && y < job.end.y && z < job.end.z){
        if( get_Field_LAP(xx, x, y, LAP) <= 0.0){

            REAL dx = get_Field_LAP(xx, x, y, 2*LAP-z) - get_Field_LAP(xx, x, y, 2*LAP-z-1);
            REAL dz = get_Field_LAP(zz, x, y, 2*LAP-z) - get_Field_LAP(zz, x, y, 2*LAP-z-1);

            REAL tmpxx = fabs(-cos(2*seta1) + sin(2*seta1)*dz/dx);
            REAL tmpzz = fabs(cos(2*seta1) + dx/dz*sin(2*seta1));

            get_Field_LAP(Ajac, x, y, z) = tmpxx*tmpzz*get_Field_LAP(Ajac, x, y, 2*LAP-z);
            get_Field_LAP(Akx, x, y, z) =  tmpzz*get_Field_LAP(Akx, x, y, 2*LAP-z);
            get_Field_LAP(Aky, x, y, z) = -tmpxx*tmpzz*get_Field_LAP(Aky, x, y, 2*LAP-z);
            get_Field_LAP(Akz, x, y, z) = -tmpxx*get_Field_LAP(Akz, x, y, 2*LAP-z);
            get_Field_LAP(Aix, x, y, z) = -tmpzz*get_Field_LAP(Aix, x, y, 2*LAP-z);
            get_Field_LAP(Aiy, x, y, z) =  tmpxx*tmpzz*get_Field_LAP(Aiy, x, y, 2*LAP-z);
            get_Field_LAP(Aiz, x, y, z) =  tmpxx*get_Field_LAP(Aiz, x, y, 2*LAP-z);
            get_Field_LAP(Asx, x, y, z) = -tmpzz*get_Field_LAP(Asx, x, y, 2*LAP-z);
            get_Field_LAP(Asy, x, y, z) =  tmpxx*tmpzz*get_Field_LAP(Asy, x, y, 2*LAP-z);
            get_Field_LAP(Asz, x, y, z) =  tmpxx*get_Field_LAP(Asz, x, y, 2*LAP-z);
            
        }else{

            REAL dx = get_Field_LAP(xx, x, y, 2*LAP-z) - get_Field_LAP(xx, x, y, 2*LAP-z-1);
            REAL dz = get_Field_LAP(zz, x, y, 2*LAP-z) - get_Field_LAP(zz, x, y, 2*LAP-z-1);

            REAL tmpxx = fabs(-cos(2*(seta1+seta2)) + sin(2*(seta1+seta2))*dz/dx);
            REAL tmpzz = fabs(cos(2*(seta1+seta2)) + dx/dz*sin(2*(seta1+seta2)));

            get_Field_LAP(Ajac, x, y, z) = tmpxx*tmpzz*get_Field_LAP(Ajac, x, y, 2*LAP-z);
            get_Field_LAP(Akx, x, y, z) =  tmpzz*get_Field_LAP(Akx, x, y, 2*LAP-z);
            get_Field_LAP(Aky, x, y, z) = -tmpxx*tmpzz*get_Field_LAP(Aky, x, y, 2*LAP-z);
            get_Field_LAP(Akz, x, y, z) = -tmpxx*get_Field_LAP(Akz, x, y, 2*LAP-z);
            get_Field_LAP(Aix, x, y, z) = -tmpzz*get_Field_LAP(Aix, x, y, 2*LAP-z);
            get_Field_LAP(Aiy, x, y, z) =  tmpxx*tmpzz*get_Field_LAP(Aiy, x, y, 2*LAP-z);
            get_Field_LAP(Aiz, x, y, z) =  tmpxx*get_Field_LAP(Aiz, x, y, 2*LAP-z);
            get_Field_LAP(Asx, x, y, z) = -tmpzz*get_Field_LAP(Asx, x, y, 2*LAP-z);
            get_Field_LAP(Asy, x, y, z) =  tmpxx*tmpzz*get_Field_LAP(Asy, x, y, 2*LAP-z);
            get_Field_LAP(Asz, x, y, z) =  tmpxx*get_Field_LAP(Asz, x, y, 2*LAP-z);
        }
    }
}

void boundary_Jac3d_kernal_z_cone_wall(REAL seta1, REAL seta2){
    if (npz == 0)
    {
        seta1 = seta1/PI;
        seta2 = seta2/PI;
        dim3 griddim , blockdim;
        cudaJobPackage job( dim3(LAP, LAP, 0) , dim3(nx_lap, ny_lap, LAP) );

        CUDA_LAUNCH(( boundary_Jac3d_kernal_z_cone_wall_kernel<<<griddim , blockdim>>>(*pAxx_d,*pAzz_d,*pAkx_d,
            *pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,*pAjac_d,seta1,seta2,job) ));
    }
}

void boundary_Jac3d_Axx()
{
    if(IF_SYMMETRY == 1){
         if (npy == 0)
        {
            dim3 griddim , blockdim;
            cudaJobPackage job( dim3(LAP , 0 , LAP) , dim3(nx_lap , LAP , nz_lap) );
            cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz );
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAxx_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAyy_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAzz_d , 1.0 , job) ));
        }
    
        if (npy == NPY0 - 1)
        {
            dim3 griddim , blockdim;
            cudaJobPackage job( dim3(LAP , ny_lap , LAP) , dim3(nx_lap , ny_2lap , nz_lap) );
            cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz );
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAxx_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAyy_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAzz_d , 1.0 , job) ));
        }
    }
}



void boundary_Jac3d_Liftbody_Ajac()
{
    if(IF_SYMMETRY == 1){
        if (npy == 0)
        {
            dim3 griddim , blockdim;
            cudaJobPackage job( dim3(LAP , 0 , LAP) , dim3(nx_lap , LAP , nz_lap) );
            cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz );
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAkx_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAky_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAkz_d , 1.0 , job) ));
    
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAix_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAiy_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAiz_d ,-1.0 , job) ));
    
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAsx_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAsy_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAsz_d , 1.0 , job) ));
    
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_l<<<griddim , blockdim>>>(*pAjac_d , 1.0 , job) ));
        }
    
        if (npy == NPY0 - 1)
        {
            dim3 griddim , blockdim;
            cudaJobPackage job( dim3(LAP , ny_lap , LAP) , dim3(nx_lap , ny_2lap , nz_lap) );
            cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , LAP , nz );
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAkx_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAky_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAkz_d , 1.0 , job) ));
    
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAix_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAiy_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAiz_d ,-1.0 , job) ));
    
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAsx_d , 1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAsy_d ,-1.0 , job) ));
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAsz_d , 1.0 , job) ));
    
            CUDA_LAUNCH(( boundary_Jac3d_kernal_y_r<<<griddim , blockdim>>>(*pAjac_d , 1.0 , job) ));
        }
    }
}


#ifdef __cplusplus
}
#endif

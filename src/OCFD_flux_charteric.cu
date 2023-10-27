#include <math.h>

#include "commen_kernel.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "parameters.h"
#include "parameters_d.h"

#include "OCFD_flux_charteric.h"
#include "OCFD_Schemes_hybrid_auto.h"
#include "OCFD_Schemes.h"
#include "OCFD_bound_Scheme.h"
#include "OCFD_warp_shuffle.h"


__device__ void put_du_character_p_kernel(dim3 flagxyz, dim3 coords, REAL tmp, REAL tmp_r, REAL tmp_l, cudaSoA du, int num, cudaField Ajac, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

	switch(flagxyz.x){
		case 1:
		case 4:
          {
		     if(flagxyz.y == 1 && flagxyz.z == 1 && coords.x == 1){
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		     }else{
                if((flagxyz.y == 1 && coords.x == 1) || (flagxyz.y == 4 && coords.x == job.end.x-job.start.x-1)){
                    atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*tmp/hx_d);
                    }else{
                    atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hx_d);
		          }
               }

          }
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP) = (tmp_r - tmp_l)/hx_d;
		break;

		case 2:
		case 5:
		if(flagxyz.y == 2 && flagxyz.z == 1 && coords.y == 1){
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
               if((flagxyz.y == 2 && coords.y == 1) || (flagxyz.y == 5 && coords.y == job.end.y-job.start.y-1)){
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*tmp/hy_d);
               }else{  
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hy_d);
		     }
          }
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP) = (tmp_r - tmp_l)/hy_d;
		break;

		case 3:
		case 6:
		if(flagxyz.y == 3 && flagxyz.z == 1 && coords.z == 1){
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
               if((flagxyz.y == 3 && coords.z == 1) || (flagxyz.y == 6 && coords.z == job.end.z-job.start.z-1)){
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*tmp/hz_d);
               }else{
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z)*(tmp_r - tmp_l)/hz_d);
		     }
          }
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP) = (tmp_r - tmp_l)/hz_d;
		break;
	}
}


__device__ void put_du_character_m_kernel(dim3 flagxyz, dim3 coords, REAL tmp, REAL tmp_r, REAL tmp_l, cudaSoA du, int num, cudaField Ajac, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

	switch(flagxyz.x){
		case 1:
		case 4:
		if(flagxyz.y == 4 && flagxyz.z == 1 && coords.x == job.end.x-job.start.x-1){
            atomicAdd(du.ptr + (x - LAP - 1 + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
               if((flagxyz.y == 1 && coords.x == 1) || (flagxyz.y == 4 && coords.x == job.end.x-job.start.x-1)){
                    atomicAdd(du.ptr + (x - LAP - 1 + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x-1, y, z)*tmp/hx_d);
               }else{
                atomicAdd(du.ptr + (x - LAP - 1 + du.pitch*(y - LAP + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x-1, y, z)*(tmp_r - tmp_l)/hx_d);
		     }
          }
		//get_Field(Ajac, x-LAP-1, y-LAP, z-LAP) = (tmp_r - tmp_l)/hx_d;
		break;

		case 2:
		case 5:
		if(flagxyz.y == 5 && flagxyz.z == 1 && coords.y == job.end.y-job.start.y-1){
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP - 1 + ny_d*(z - LAP + (num)*nz_d))), 0);
		}else{
               if((flagxyz.y == 3 && coords.y == 1) || (flagxyz.y == 5 && coords.y == job.end.y-job.start.y-1)){
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP - 1 + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y-1, z)*tmp/hy_d);
               }else{
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP - 1 + ny_d*(z - LAP + (num)*nz_d))), -get_Field_LAP(Ajac, x, y-1, z)*(tmp_r - tmp_l)/hy_d);
		     }
          }
		//get_Field(Ajac, x-LAP, y-LAP-1, z-LAP) = (tmp_r - tmp_l)/hy_d;
		break;

		case 3:
		case 6:
		if(flagxyz.y == 6 && flagxyz.z == 1 && coords.z == job.end.z-job.start.z-1){
            atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP - 1 + (num)*nz_d))), 0);
		}else{
               if((flagxyz.y == 3 && coords.z == 1) || (flagxyz.y == 6 && coords.z == job.end.z-job.start.z-1)){
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP - 1 + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z-1)*tmp/hz_d);
               }else{
                atomicAdd(du.ptr + (x - LAP + du.pitch*(y - LAP + ny_d*(z - LAP - 1 + (num)*nz_d))), -get_Field_LAP(Ajac, x, y, z-1)*(tmp_r - tmp_l)/hz_d);
		     }
          }
		//get_Field(Ajac, x-LAP, y-LAP, z-LAP-1) = (tmp_r - tmp_l)/hz_d;
		break;
	}
}


__device__ REAL OCFD_bound_character_kernel_p(dim3 flagxyzb, dim3 coords, REAL *stencil, cudaJobPackage job){

     REAL tmp;

	switch(flagxyzb.y){
		case 1:
		{
			if(coords.x == 1){

                    tmp = (stencil[1] - stencil[0]);
                    //tmp = (-11.0*stencil[0] + 18.0*stencil[1] - 9.0*stencil[2] + 2.0*stencil[3])/6.0;
                    //tmp = (-2.0*stencil[0] - 3.0*stencil[1] + 6.0*stencil[2] - stencil[3])/6.0;

			     return tmp;

			}
		}
		break;

		case 2:
		{
			if(coords.y == 1){

                    tmp = (stencil[1] - stencil[0]);
                    //tmp = (-11.0*stencil[0] + 18.0*stencil[1] - 9.0*stencil[2] + 2.0*stencil[3])/6.0;
                    //tmp = (-2.0*stencil[0] - 3.0*stencil[1] + 6.0*stencil[2] - stencil[3])/6.0;

				return tmp;

			}
		}
		break;

		case 3:
		{
			if(coords.z == 1){

                    tmp = (stencil[1] - stencil[0]);
                    //tmp = (-11.0*stencil[0] + 18.0*stencil[1] - 9.0*stencil[2] + 2.0*stencil[3])/6.0;
                    //tmp = (-2.0*stencil[0] - 3.0*stencil[1] - 6.0*stencil[2] - stencil[3])/6.0;

				return tmp;

			}
		}
		break;

		case 4:
		{
			if(coords.x == job.end.x-job.start.x-1){

                    REAL tmp_r = stencil[0] + 0.5*minmod2(stencil[0] - stencil[-1], stencil[0] - stencil[-1]);
                    REAL tmp_l = stencil[-1] + 0.5*minmod2(stencil[0] - stencil[-1], stencil[-1] - stencil[-2]);

                    //tmp = (11.0*stencil[0] - 18.0*stencil[-1] + 9.0*stencil[-2] - 2.0*stencil[-3])/6.0;
                    tmp = tmp_r - tmp_l;

				return tmp;

			}
		}
		break;

		case 5:
		{
			if(coords.y == job.end.y-job.start.y-1){

                    //tmp = (11.0*stencil[0] - 18.0*stencil[-1] + 9.0*stencil[-2] - 2.0*stencil[-3])/6.0;
                    REAL tmp_r = stencil[0] + 0.5*minmod2(stencil[0] - stencil[-1], stencil[0] - stencil[-1]);
                    REAL tmp_l = stencil[-1] + 0.5*minmod2(stencil[0] - stencil[-1], stencil[-1] - stencil[-2]);

                    tmp = tmp_r - tmp_l;

				return tmp;

			}
		}
		break;

		case 6:
		{
			if(coords.z == job.end.z-job.start.z-1){

                    //tmp = (11.0*stencil[0] - 18.0*stencil[-1] + 9.0*stencil[-2] - 2.0*stencil[-3])/6.0;
                    REAL tmp_r = stencil[0] + 0.5*minmod2(stencil[0] - stencil[-1], stencil[0] - stencil[-1]);
                    REAL tmp_l = stencil[-1] + 0.5*minmod2(stencil[0] - stencil[-1], stencil[-1] - stencil[-2]);

                    tmp = tmp_r - tmp_l;

				return tmp;

			}
		}
		break;
	}

     return 0.0;
}



__device__ REAL OCFD_bound_character_kernel_m(dim3 flagxyzb, dim3 coords, REAL *stencil, cudaJobPackage job){

     REAL tmp;

	switch(flagxyzb.y){
		case 1:
		{
			if(coords.x == 1){

                    //tmp = (-11.0*stencil[-1] + 18.0*stencil[0] - 9.0*stencil[1] + 2.0*stencil[2])/6.0;
                    REAL tmp_r = stencil[0] - 0.5*minmod2(stencil[1] - stencil[0], stencil[0] - stencil[-1]);
                    REAL tmp_l = stencil[-1] - 0.5*minmod2(stencil[0] - stencil[-1], stencil[0] - stencil[-1]);

                    tmp = tmp_r - tmp_l;

				return tmp;

			}
		}
		break;

		case 2:
		{
			if(coords.y == 1){

                    //tmp = (-11.0*stencil[-1] + 18.0*stencil[0] - 9.0*stencil[1] + 2.0*stencil[2])/6.0;
                    REAL tmp_r = stencil[0] - 0.5*minmod2(stencil[1] - stencil[0], stencil[0] - stencil[-1]);
                    REAL tmp_l = stencil[-1] - 0.5*minmod2(stencil[0] - stencil[-1], stencil[0] - stencil[-1]);

                    tmp = tmp_r - tmp_l;

				return tmp;

			}
		}
		break;

		case 3:
		{
			if(coords.z == 1){
				
                    //tmp = (-11.0*stencil[-1] + 18.0*stencil[0] - 9.0*stencil[1] + 2.0*stencil[2])/6.0;
                    REAL tmp_r = stencil[0] - 0.5*minmod2(stencil[1] - stencil[0], stencil[0] - stencil[-1]);
                    REAL tmp_l = stencil[-1] - 0.5*minmod2(stencil[0] - stencil[-1], stencil[0] - stencil[-1]);

                    tmp = tmp_r - tmp_l;

				return tmp;

			}
		}
		break;

		case 4:
		{
			
			if(coords.x == job.end.x-job.start.x-1){

                    //tmp = (11.0*stencil[-1] - 18.0*stencil[-2] + 9.0*stencil[-3] - 2.0*stencil[-4])/6.0;
                    tmp = (stencil[-1] - stencil[-2]);

				return tmp;

			}
		}
		break;

		case 5:
		{

			if(coords.y == job.end.y-job.start.y-1){

                    //tmp = (11.0*stencil[-1] - 18.0*stencil[-2] + 9.0*stencil[-3] - 2.0*stencil[-4])/6.0;
                    tmp = (stencil[-1] - stencil[-2]);

				return tmp;

			}
		}
		break;

		case 6:
		{

			if(coords.z == job.end.z-job.start.z-1){

                    //tmp = (11.0*stencil[-1] - 18.0*stencil[-2] + 9.0*stencil[-3] - 2.0*stencil[-4])/6.0;
                    tmp = (stencil[-1] - stencil[-2]);

				return tmp;

			}
		}
		break;
	}

     return 0.0;
}

__device__ void get_para_charteric_p_kernel(
    int flagxyz,
    dim3 coords,
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    REAL *para_ch,
    cudaJobPackage job
){
    unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

     REAL u1, v1, w1, c1, a1, a2, a3, ss, n1, n2, n3, l1, l2, l3, m1, m2, m3, KK;

    switch(flagxyz){
         case 1:
         case 4:
         {
              u1 = (get_Field_LAP(u,  x, y, z) + get_Field_LAP(u,  x+1, y, z))*0.5;
              v1 = (get_Field_LAP(v,  x, y, z) + get_Field_LAP(v,  x+1, y, z))*0.5;
              w1 = (get_Field_LAP(w,  x, y, z) + get_Field_LAP(w,  x+1, y, z))*0.5;
              c1 = (get_Field_LAP(cc, x, y, z) + get_Field_LAP(cc, x+1, y, z))*0.5;
              a1 = (get_Field_LAP(Ax, x, y, z) + get_Field_LAP(Ax, x+1, y, z))*0.5;
              a2 = (get_Field_LAP(Ay, x, y, z) + get_Field_LAP(Ay, x+1, y, z))*0.5;
              a3 = (get_Field_LAP(Az, x, y, z) + get_Field_LAP(Az, x+1, y, z))*0.5;
         }
         break;

         case 2:
         case 5:
         {
              u1 = (get_Field_LAP(u,  x, y, z) + get_Field_LAP(u,  x, y+1, z))*0.5;
              v1 = (get_Field_LAP(v,  x, y, z) + get_Field_LAP(v,  x, y+1, z))*0.5;
              w1 = (get_Field_LAP(w,  x, y, z) + get_Field_LAP(w,  x, y+1, z))*0.5;
              c1 = (get_Field_LAP(cc, x, y, z) + get_Field_LAP(cc, x, y+1, z))*0.5;
              a1 = (get_Field_LAP(Ax, x, y, z) + get_Field_LAP(Ax, x, y+1, z))*0.5;
              a2 = (get_Field_LAP(Ay, x, y, z) + get_Field_LAP(Ay, x, y+1, z))*0.5;
              a3 = (get_Field_LAP(Az, x, y, z) + get_Field_LAP(Az, x, y+1, z))*0.5;
         }
         break;

         case 3:
         case 6:
         {
              u1 = (get_Field_LAP(u,  x, y, z) + get_Field_LAP(u,  x, y, z+1))*0.5;
              v1 = (get_Field_LAP(v,  x, y, z) + get_Field_LAP(v,  x, y, z+1))*0.5;
              w1 = (get_Field_LAP(w,  x, y, z) + get_Field_LAP(w,  x, y, z+1))*0.5;
              c1 = (get_Field_LAP(cc, x, y, z) + get_Field_LAP(cc, x, y, z+1))*0.5;
              a1 = (get_Field_LAP(Ax, x, y, z) + get_Field_LAP(Ax, x, y, z+1))*0.5;
              a2 = (get_Field_LAP(Ay, x, y, z) + get_Field_LAP(Ay, x, y, z+1))*0.5;
              a3 = (get_Field_LAP(Az, x, y, z) + get_Field_LAP(Az, x, y, z+1))*0.5;
         }
         break;
    }

	ss = sqrt(a1*a1 + a2*a2 + a3*a3);
    n1 = a1/ss;
    n2 = a2/ss;
    n3 = a3/ss;

    if(fabs(n3) <= fabs(n2)){
        ss = sqrt(n1*n1 + n2*n2);
        l1 = -n2/ss;
        l2 = n1/ss;
        l3 = 0.0;
    }else{
        ss = sqrt(n1*n1 + n3*n3);
        l1 = -n3/ss;
        l2 = 0.0;
        l3 = n1/ss;
    }

    m1 = n2*l3 - n3*l2;
    m2 = n3*l1 - n1*l3;
    m3 = n1*l2 - n2*l1;

    KK = (Gamma_d - 1.0)/(c1*c1);

    para_ch[0] = u1;
    para_ch[1] = v1;
    para_ch[2] = w1;
    para_ch[3] = c1;
    para_ch[4] = n1;
    para_ch[5] = n2;
    para_ch[6] = n3;
    para_ch[7] = l1;
    para_ch[8] = l2;
    para_ch[9] = l3;
    para_ch[10] = m1;
    para_ch[11] = m2;
    para_ch[12] = m3;
    para_ch[13] = KK;

}


__device__ void get_para_charteric_m_kernel(
    int flagxyz,
    dim3 coords,
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    REAL *para_ch,
    cudaJobPackage job
){
    unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

    REAL u1, v1, w1, c1, a1, a2, a3, ss, n1, n2, n3, l1, l2, l3, m1, m2, m3, KK;

    switch(flagxyz){
         case 1:
         case 4:
         {
              u1 = (get_Field_LAP(u,  x-1, y, z) + get_Field_LAP(u,  x, y, z))*0.5;
              v1 = (get_Field_LAP(v,  x-1, y, z) + get_Field_LAP(v,  x, y, z))*0.5;
              w1 = (get_Field_LAP(w,  x-1, y, z) + get_Field_LAP(w,  x, y, z))*0.5;
              c1 = (get_Field_LAP(cc, x-1, y, z) + get_Field_LAP(cc, x, y, z))*0.5;
              a1 = (get_Field_LAP(Ax, x-1, y, z) + get_Field_LAP(Ax, x, y, z))*0.5;
              a2 = (get_Field_LAP(Ay, x-1, y, z) + get_Field_LAP(Ay, x, y, z))*0.5;
              a3 = (get_Field_LAP(Az, x-1, y, z) + get_Field_LAP(Az, x, y, z))*0.5;
         }
         break;

         case 2:
         case 5:
         {
              u1 = (get_Field_LAP(u,  x, y-1, z) + get_Field_LAP(u,  x, y, z))*0.5;
              v1 = (get_Field_LAP(v,  x, y-1, z) + get_Field_LAP(v,  x, y, z))*0.5;
              w1 = (get_Field_LAP(w,  x, y-1, z) + get_Field_LAP(w,  x, y, z))*0.5;
              c1 = (get_Field_LAP(cc, x, y-1, z) + get_Field_LAP(cc, x, y, z))*0.5;
              a1 = (get_Field_LAP(Ax, x, y-1, z) + get_Field_LAP(Ax, x, y, z))*0.5;
              a2 = (get_Field_LAP(Ay, x, y-1, z) + get_Field_LAP(Ay, x, y, z))*0.5;
              a3 = (get_Field_LAP(Az, x, y-1, z) + get_Field_LAP(Az, x, y, z))*0.5;
         }
         break;

         case 3:
         case 6:
         {
              u1 = (get_Field_LAP(u,  x, y, z-1) + get_Field_LAP(u,  x, y, z))*0.5;
              v1 = (get_Field_LAP(v,  x, y, z-1) + get_Field_LAP(v,  x, y, z))*0.5;
              w1 = (get_Field_LAP(w,  x, y, z-1) + get_Field_LAP(w,  x, y, z))*0.5;
              c1 = (get_Field_LAP(cc, x, y, z-1) + get_Field_LAP(cc, x, y, z))*0.5;
              a1 = (get_Field_LAP(Ax, x, y, z-1) + get_Field_LAP(Ax, x, y, z))*0.5;
              a2 = (get_Field_LAP(Ay, x, y, z-1) + get_Field_LAP(Ay, x, y, z))*0.5;
              a3 = (get_Field_LAP(Az, x, y, z-1) + get_Field_LAP(Az, x, y, z))*0.5;
         }
         break;
    }

	ss = sqrt(a1*a1 + a2*a2 + a3*a3);
    n1 = a1/ss;
    n2 = a2/ss;
    n3 = a3/ss;

    if(fabs(n3) <= fabs(n2)){
        ss = sqrt(n1*n1 + n2*n2);
        l1 = -n2/ss;
        l2 = n1/ss;
        l3 = 0.0;
    }else{
        ss = sqrt(n1*n1 + n3*n3);
        l1 = -n3/ss;
        l2 = 0.0;
        l3 = n1/ss;
    }

    m1 = n2*l3 - n3*l2;
    m2 = n3*l1 - n1*l3;
    m3 = n1*l2 - n2*l1;

    KK = (Gamma_d - 1.0)/(c1*c1);

    para_ch[0] = u1;
    para_ch[1] = v1;
    para_ch[2] = w1;
    para_ch[3] = c1;
    para_ch[4] = n1;
    para_ch[5] = n2;
    para_ch[6] = n3;
    para_ch[7] = l1;
    para_ch[8] = l2;
    para_ch[9] = l3;
    para_ch[10] = m1;
    para_ch[11] = m2;
    para_ch[12] = m3;
    para_ch[13] = KK;

}

__device__ void get_du_charteric_p_kernel(dim3 flagxyzb, dim3 coords, cudaSoA du, REAL *stencil, cudaField Ajac, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

     switch(flagxyzb.x){
	     case 1:
	     case 4:
	     if(flagxyzb.x == 4 && coords.x == 1 && flagxyzb.z == 1){
                 for(int i = 0; i < 5; i++){
	          	get_SoA(du, x-LAP, y-LAP, z-LAP, i) += 0;
                 }
	     }else{ 
                 for(int i = 0; i < 5; i++){
	          	get_SoA(du, x-LAP, y-LAP, z-LAP, i) += -get_Field_LAP(Ajac, x, y, z)*stencil[i]/hx_d;
	          }
            }
	     break;

	     case 2:
	     case 5:
	     if(flagxyzb.x == 5 && coords.y == 1 && flagxyzb.z == 1){
               for(int i = 0; i < 5; i++){
                    get_SoA(du, x-LAP, y-LAP, z-LAP, i) += 0;
               }
	     }else{ 
               for(int i = 0; i < 5; i++){
                    get_SoA(du, x-LAP, y-LAP, z-LAP, i) += -get_Field_LAP(Ajac, x, y, z)*stencil[i]/hy_d;
               }
	     }
	     break;

	     case 3:
	     case 6:
	     if(flagxyzb.x == 6 && coords.z == 1 && flagxyzb.z == 1){
               for(int i = 0; i < 5; i++){
                    get_SoA(du, x-LAP, y-LAP, z-LAP, i) += 0;
               }
	     }else{ 
               for(int i = 0; i < 5; i++){
                    get_SoA(du, x-LAP, y-LAP, z-LAP, i) += -get_Field_LAP(Ajac, x, y, z)*stencil[i]/hz_d;
               }
	     }
	     break;
	}
}


__device__ void get_du_charteric_m_kernel(dim3 flagxyzb, dim3 coords, cudaSoA du, REAL *stencil, cudaField Ajac, cudaJobPackage job){
	unsigned int x = coords.x + job.start.x;
	unsigned int y = coords.y + job.start.y;
	unsigned int z = coords.z + job.start.z;

    switch(flagxyzb.x){
	    case 1:
	    case 4:
	    if(flagxyzb.y == 4 && coords.x == job.end.x-job.start.x-1 && flagxyzb.z == 1){
                for(int i = 0; i < 5; i++){
	         	get_SoA(du, x-LAP-1, y-LAP, z-LAP, i) += 0;
                }
	    }else{ 
                for(int i = 0; i < 5; i++){
	         	get_SoA(du, x-LAP-1, y-LAP, z-LAP, i) += -get_Field_LAP(Ajac, x-1, y, z)*stencil[i]/hx_d;
	         }
           }
	    break;
	    case 2:
	    case 5:
	    if(flagxyzb.y == 5 && coords.y == job.end.y-job.start.y-1 && flagxyzb.z == 1){
              for(int i = 0; i < 5; i++){
                   get_SoA(du, x-LAP, y-LAP-1, z-LAP, i) += 0;
              }
	    }else{ 
              for(int i = 0; i < 5; i++){
                   get_SoA(du, x-LAP, y-LAP-1, z-LAP, i) += -get_Field_LAP(Ajac, x, y-1, z)*stencil[i]/hy_d;
              }
	    }
	    break;

	    case 3:
	    case 6:
	    if(flagxyzb.y == 6 && coords.z == job.end.z-job.start.z-1 && flagxyzb.z == 1){
              for(int i = 0; i < 5; i++){
                   get_SoA(du, x-LAP, y-LAP, z-LAP-1, i) += 0;
              }
	    }else{ 
              for(int i = 0; i < 5; i++){
                   get_SoA(du, x-LAP, y-LAP, z-LAP-1, i) += -get_Field_LAP(Ajac, x, y, z-1)*stencil[i]/hz_d;
              }
	    }
	    break;
	}
}


__device__ void flux_charteric_ptoc_kernel(
	REAL *stencil_ch,
    REAL *para_ch)
{
	//Transform variables to character space

    REAL u1 = para_ch[0];
    REAL v1 = para_ch[1];
    REAL w1 = para_ch[2];
    REAL c1 = para_ch[3];
    REAL n1 = para_ch[4];
    REAL n2 = para_ch[5];
    REAL n3 = para_ch[6];
    REAL l1 = para_ch[7];
    REAL l2 = para_ch[8];
    REAL l3 = para_ch[9];
    REAL m1 = para_ch[10];
    REAL m2 = para_ch[11];
    REAL m3 = para_ch[12];
    REAL KK = para_ch[13];
    REAL X1 = 1.0/(2.0*c1);;

    REAL un = u1*n1 + v1*n2 + w1*n3;
    REAL v2 = (u1*u1 + v1*v1 + w1*w1)*0.5;

    //====================S=L（Left Characteristic Matrix)
    REAL S11 = 1.0 - KK*v2;
    REAL S12 = KK*u1;
    REAL S13 = KK*v1;
    REAL S14 = KK*w1;
    REAL S15 = -KK;

    REAL S21 = -(u1*l1 + v1*l2 + w1*l3);
    REAL S22 = l1;
    REAL S23 = l2;
    REAL S24 = l3;
    REAL S25 = 0.0;

    REAL S31 = -(u1*m1 + v1*m2 + w1*m3);
    REAL S32 = n2*l3 - n3*l2;
    REAL S33 = n3*l1 - n1*l3;
    REAL S34 = n1*l2 - n2*l1;
    REAL S35 = 0.0;

    REAL S41 = KK*v2*0.5 + X1*un; 
    REAL S42 = -X1*n1 - KK*u1*0.5;
    REAL S43 = -X1*n2 - KK*v1*0.5;
    REAL S44 = -X1*n3 - KK*w1*0.5;
    REAL S45 = KK*0.5;

    REAL S51 = S41 - 2*X1*un;
    REAL S52 = 2*X1*n1 + S42;
    REAL S53 = 2*X1*n2 + S43;
    REAL S54 = 2*X1*n3 + S44;
    REAL S55 = S45;

    m1 = S11*stencil_ch[0] + S12*stencil_ch[1] + S13*stencil_ch[2] + S14*stencil_ch[3] + S15*stencil_ch[4];
    m2 = S21*stencil_ch[0] + S22*stencil_ch[1] + S23*stencil_ch[2] + S24*stencil_ch[3] + S25*stencil_ch[4];
    m3 = S31*stencil_ch[0] + S32*stencil_ch[1] + S33*stencil_ch[2] + S34*stencil_ch[3] + S35*stencil_ch[4];
    l1 = S41*stencil_ch[0] + S42*stencil_ch[1] + S43*stencil_ch[2] + S44*stencil_ch[3] + S45*stencil_ch[4];
    l2 = S51*stencil_ch[0] + S52*stencil_ch[1] + S53*stencil_ch[2] + S54*stencil_ch[3] + S55*stencil_ch[4];

    stencil_ch[0] = m1;
    stencil_ch[1] = m2;
    stencil_ch[2] = m3;
    stencil_ch[3] = l1;
    stencil_ch[4] = l2;
        
}

__device__ void flux_charteric_ctop_kernel(
	REAL *stencil,
    REAL *para_ch)
{
	//Transform variables to character space

     REAL u1 = para_ch[0];
     REAL v1 = para_ch[1];
     REAL w1 = para_ch[2];
     REAL c1 = para_ch[3];
     REAL n1 = para_ch[4];
     REAL n2 = para_ch[5];
     REAL n3 = para_ch[6];
     REAL l1 = para_ch[7];
     REAL l2 = para_ch[8];
     REAL l3 = para_ch[9];
     REAL m1 = para_ch[10];
     REAL m2 = para_ch[11];
     REAL m3 = para_ch[12];
     REAL KK = para_ch[13];

    //====================S=L（Left Characteristic Matrix)
     REAL S11 = 1.0;
     REAL S12 = 0.0;
     REAL S13 = 0.0;
     REAL S14 = 1.0;
     REAL S15 = 1.0;

     REAL S21 = u1;
     REAL S22 = l1;
     REAL S23 = n2*l3 - n3*l2;
     REAL S24 = u1 - c1*n1;
     REAL S25 = 2*u1 - S24;

     REAL S31 = v1;
     REAL S32 = l2;
     REAL S33 = n3*l1 - n1*l3;
     REAL S34 = v1 - c1*n2;
     REAL S35 = 2*v1 - S34;

     REAL S41 = w1; 
     REAL S42 = l3;
     REAL S43 = n1*l2 - n2*l1;
     REAL S44 = w1 - c1*n3;
     REAL S45 = 2*w1 - S44;

     REAL S51 = (u1*u1 + v1*v1 + w1*w1)*0.5;
     REAL H = S51 + 1.0/KK;
     REAL S52 = u1*l1 + v1*l2 + w1*l3;
     REAL S53 = u1*m1 + v1*m2 + w1*m3;
     REAL S54 = H - c1*(u1*n1 + v1*n2 + w1*n3);
     REAL S55 = 2*H - S54;

     m1 = S11*stencil[0] + S12*stencil[1] + S13*stencil[2] + S14*stencil[3] + S15*stencil[4];
     m2 = S21*stencil[0] + S22*stencil[1] + S23*stencil[2] + S24*stencil[3] + S25*stencil[4];
     m3 = S31*stencil[0] + S32*stencil[1] + S33*stencil[2] + S34*stencil[3] + S35*stencil[4];
     l1 = S41*stencil[0] + S42*stencil[1] + S43*stencil[2] + S44*stencil[3] + S45*stencil[4];
     l2 = S51*stencil[0] + S52*stencil[1] + S53*stencil[2] + S54*stencil[3] + S55*stencil[4];

     stencil[0] = m1;
     stencil[1] = m2;
     stencil[2] = m3;
     stencil[3] = l1;
     stencil[4] = l2;

}

__global__ void OCFD_weno7_SYMBO_character_P_kernel(
    int WENO_LMT_FLAG, 
    dim3 flagxyzb, 
    cudaSoA f, 
    cudaSoA du, 
    cudaField Ajac, 
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    cudaJobPackage job)
{
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[40];
    REAL para_ch[14];
    REAL tmp[5], tmp_r, tmp_l;

	int flag; 
    int ia1 = -3; int ib1 = 4;

	for(int i = 0; i < 5; i++){

		flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[8*i], ia1, ib1, sort, job);

        tmp[i] = OCFD_bound_character_kernel_p(flagxyzb, coords, &stencil[8*i-ia1], job);
    }



	if(flag != 0){

        {
            REAL stencil_ch[40];

            get_para_charteric_p_kernel(flagxyzb.x, coords, u, v, w, cc, Ax, Ay, Az, &para_ch[0], job);

            for(int j = 0; j < 8; j++){
                for(int i = 0; i < 5; i++){
                     stencil_ch[5*j+i] = stencil[8*i+j];//转置
                }

                flux_charteric_ptoc_kernel(&stencil_ch[5*j], &para_ch[0]);

                for(int i = 0; i < 5; i++){
                     stencil[8*i+j] = stencil_ch[5*j+i];//转置
                }
            }
        }//特征重构

        for(int i = 0; i < 5; i++){

			flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[8*i], ia1, ib1, job);

			if(flag != 0) tmp_r = OCFD_weno7_SYMBO_kernel_P(WENO_LMT_FLAG, &stencil[8*i]);

            stencil[i] = tmp_r;

		}

        flux_charteric_ctop_kernel(&stencil[0], &para_ch[0]);

        for(int i = 0; i < 5; i++){

            tmp_r = stencil[i];
 
            tmp_l = __shfl_up_double(tmp_r, 1, warpSize);
 
            if(threadIdx.x != 0) put_du_character_p_kernel(flagxyzb, coords, tmp[i], tmp_r, tmp_l, du, i, Ajac, job);
 
        }
	}
}



__global__ void OCFD_weno7_SYMBO_character_M_kernel(
    int WENO_LMT_FLAG, 
    dim3 flagxyzb, 
    cudaSoA f, 
    cudaSoA du, 
    cudaField Ajac, 
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    cudaJobPackage job)
{
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[40];
    REAL para_ch[14];
    REAL tmp[5], tmp_r, tmp_l; 

    int flag;
	int ia1 = -4; int ib1 = 3;

    for(int i = 0; i < 5; i++){

		flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[8*i], ia1, ib1, sort, job);

        tmp[i] = OCFD_bound_character_kernel_m(flagxyzb, coords, &stencil[8*i-ia1], job);
    }


    if(flag != 0){

        {
            REAL stencil_ch[40];
            get_para_charteric_m_kernel(flagxyzb.x, coords, u, v, w, cc, Ax, Ay, Az, &para_ch[0], job);

            for(int j = 0; j < 8; j++){
                for(int i = 0; i < 5; i++){
                     stencil_ch[5*j+i] = stencil[8*i+j];//转置
                }

                flux_charteric_ptoc_kernel(&stencil_ch[5*j], &para_ch[0]);

                for(int i = 0; i < 5; i++){
                     stencil[8*i+j] = stencil_ch[5*j+i];//转置
                }
            }
        }//特征重构

        for(int i = 0; i < 5; i++){

		    flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[8*i], ia1, ib1, job); 

		    if(flag != 0) tmp_r = OCFD_weno7_SYMBO_kernel_M(WENO_LMT_FLAG, &stencil[8*i]);

            stencil[i] = tmp_r;

		}

        flux_charteric_ctop_kernel(&stencil[0], &para_ch[0]);

        for(int i = 0; i < 5; i++){

            tmp_r = stencil[i];
 
            tmp_l = __shfl_up_double(tmp_r, 1, warpSize);
 
            if(threadIdx.x != 0) put_du_character_m_kernel(flagxyzb, coords, tmp[i], tmp_r, tmp_l, du, i, Ajac, job);
 
        }
	}
}



__global__ void OCFD_HybridAuto_character_P_kernel(
    dim3 flagxyzb, 
    cudaSoA f, 
    cudaSoA du, 
    cudaField Ajac, 
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    cudaField_int scheme,
    cudaJobPackage job)
{
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[40];
    REAL para_ch[14];
    REAL tmp[5], tmp_r, tmp_l;
    int Hyscheme_flag;

	int flag; 
    int ia1 = -3; int ib1 = 4;

	for(int i = 0; i < 5; i++){

		flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[8*i], ia1, ib1, sort, job);

        tmp[i] = OCFD_bound_character_kernel_p(flagxyzb, coords, &stencil[8*i-ia1], job);
    }



	if(flag != 0){

        Hyscheme_flag = get_Hyscheme_flag_p_kernel(flagxyzb.x, coords, scheme, job);

        {
           REAL stencil_ch[40];

           get_para_charteric_p_kernel(flagxyzb.x, coords, u, v, w, cc, Ax, Ay, Az, &para_ch[0], job);

           for(int j = 0; j < 8; j++){
                for(int i = 0; i < 5; i++){
                     stencil_ch[5*j+i] = stencil[8*i+j];//转置
                }

                flux_charteric_ptoc_kernel(&stencil_ch[5*j], &para_ch[0]);

                for(int i = 0; i < 5; i++){
                     stencil[8*i+j] = stencil_ch[5*j+i];//转置
                }
            }
        }//特征重构

        for(int i = 0; i < 5; i++){

			flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[8*i], ia1, ib1, job);

            if(flag != 0){
                if(Hyscheme_flag == 1){
                     tmp_r = OCFD_OMP6_kernel_P(0, &stencil[8*i]);
                }else if(Hyscheme_flag == 2){
                     tmp_r = OCFD_weno7_kernel_P(&stencil[8*i]);
                }else{
                     tmp_r = OCFD_NND2_kernel_P(&stencil[8*i+2]);
                }
            }
                

            stencil[i] = tmp_r;

		}

        flux_charteric_ctop_kernel(&stencil[0], &para_ch[0]);

        for(int i = 0; i < 5; i++){

            tmp_r = stencil[i];
 
            tmp_l = __shfl_up_double(tmp_r, 1, warpSize);
 
            if(threadIdx.x != 0) put_du_character_p_kernel(flagxyzb, coords, tmp[i], tmp_r, tmp_l, du, i, Ajac, job);
 
        }
	}
}


__global__ void OCFD_HybridAuto_character_P_Jameson_kernel(
    dim3 flagxyzb, 
    cudaSoA f, 
    cudaSoA du, 
    cudaField Ajac, 
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    cudaField_int scheme,
    cudaJobPackage job)
{
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[40];
    REAL para_ch[14];
    REAL tmp[5], tmp_r, tmp_l;
    int Hyscheme_flag;

	int flag; 
    int ia1 = -3; int ib1 = 4;

	for(int i = 0; i < 5; i++){

		flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[8*i], ia1, ib1, sort, job);

        tmp[i] = OCFD_bound_character_kernel_p(flagxyzb, coords, &stencil[8*i-ia1], job);
    }



	if(flag != 0){

        Hyscheme_flag = get_Hyscheme_flag_p_kernel(flagxyzb.x, coords, scheme, job);

        if(Hyscheme_flag != 1){

            {
                REAL stencil_ch[40];

                get_para_charteric_p_kernel(flagxyzb.x, coords, u, v, w, cc, Ax, Ay, Az, &para_ch[0], job);

                for(int j = 0; j < 8; j++){
                     for(int i = 0; i < 5; i++){
                          stencil_ch[5*j+i] = stencil[8*i+j];//转置
                     }

                     flux_charteric_ptoc_kernel(&stencil_ch[5*j], &para_ch[0]);

                     for(int i = 0; i < 5; i++){
                          stencil[8*i+j] = stencil_ch[5*j+i];//转置
                     }
                }
            }//特征重构
        }

        for(int i = 0; i < 5; i++){

			flag = OCFD_bound_scheme_kernel_p(&tmp_r, flagxyzb, coords, &stencil[8*i], ia1, ib1, job);

            if(flag != 0){
                if(Hyscheme_flag == 1){
                     tmp_r = OCFD_UP7_kernel_P(&stencil[8*i]);
                }else if(Hyscheme_flag == 2){
                     tmp_r = OCFD_weno7_kernel_P(&stencil[8*i]);
                }else{
                     tmp_r = OCFD_weno5_kernel_P(&stencil[8*i+1]);
                }
            }
                

            stencil[i] = tmp_r;

		}

        if(Hyscheme_flag != 1) flux_charteric_ctop_kernel(&stencil[0], &para_ch[0]);

        for(int i = 0; i < 5; i++){

            tmp_r = stencil[i];
 
            tmp_l = __shfl_up_double(tmp_r, 1, warpSize);
 
            if(threadIdx.x != 0) put_du_character_p_kernel(flagxyzb, coords, tmp[i], tmp_r, tmp_l, du, i, Ajac, job);
 
        }
	}
}



__global__ void OCFD_HybridAuto_character_M_kernel(
    dim3 flagxyzb, 
    cudaSoA f, 
    cudaSoA du, 
    cudaField Ajac, 
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    cudaField_int scheme,
    cudaJobPackage job)
{
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[40];
    REAL para_ch[14];
    REAL tmp[5], tmp_r, tmp_l;
    int Hyscheme_flag; 

    int flag;
	int ia1 = -4; int ib1 = 3;

    for(int i = 0; i < 5; i++){

		flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[8*i], ia1, ib1, sort, job);

        tmp[i] = OCFD_bound_character_kernel_m(flagxyzb, coords, &stencil[8*i-ia1], job);
    }


    if(flag != 0){

        Hyscheme_flag = get_Hyscheme_flag_m_kernel(flagxyzb.x, coords, scheme, job);

        {
            REAL stencil_ch[40];
            get_para_charteric_m_kernel(flagxyzb.x, coords, u, v, w, cc, Ax, Ay, Az, &para_ch[0], job);

            for(int j = 0; j < 8; j++){
                for(int i = 0; i < 5; i++){
                     stencil_ch[5*j+i] = stencil[8*i+j];//转置
                }

                flux_charteric_ptoc_kernel(&stencil_ch[5*j], &para_ch[0]);

                for(int i = 0; i < 5; i++){
                     stencil[8*i+j] = stencil_ch[5*j+i];//转置
                }
            }
        }//特征重构

        for(int i = 0; i < 5; i++){

		    flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[8*i], ia1, ib1, job); 

            if(flag != 0){
                if(Hyscheme_flag == 1){
                     tmp_r = OCFD_OMP6_kernel_M(0, &stencil[8*i]);
                }else if(Hyscheme_flag == 2){
                     tmp_r = OCFD_weno7_kernel_M(&stencil[8*i+1]);
                }else{
                     tmp_r = OCFD_NND2_kernel_M(&stencil[8*i+3]);
                }
            }

            stencil[i] = tmp_r;

		}

        flux_charteric_ctop_kernel(&stencil[0], &para_ch[0]);

        for(int i = 0; i < 5; i++){

            tmp_r = stencil[i];
 
            tmp_l = __shfl_up_double(tmp_r, 1, warpSize);
 
            if(threadIdx.x != 0) put_du_character_m_kernel(flagxyzb, coords, tmp[i], tmp_r, tmp_l, du, i, Ajac, job);
 
        }
	}
}



__global__ void OCFD_HybridAuto_character_M_Jameson_kernel(
    dim3 flagxyzb, 
    cudaSoA f, 
    cudaSoA du, 
    cudaField Ajac, 
    cudaField u,
	cudaField v,
	cudaField w,
	cudaField cc,
	cudaField Ax,
	cudaField Ay,
	cudaField Az,
    cudaField_int scheme,
    cudaJobPackage job)
{
    extern __shared__ REAL sort[];
	dim3 coords;
	REAL stencil[40];
    REAL para_ch[14];
    REAL tmp[5], tmp_r, tmp_l;
    int Hyscheme_flag; 

    int flag;
	int ia1 = -4; int ib1 = 3;

    for(int i = 0; i < 5; i++){

		flag = get_data_kernel(flagxyzb.x, &coords, f, i, &stencil[8*i], ia1, ib1, sort, job);

        tmp[i] = OCFD_bound_character_kernel_m(flagxyzb, coords, &stencil[8*i-ia1], job);
    }


    if(flag != 0){

        Hyscheme_flag = get_Hyscheme_flag_m_kernel(flagxyzb.x, coords, scheme, job);

        if(Hyscheme_flag != 1){
            {
                REAL stencil_ch[40];
                get_para_charteric_m_kernel(flagxyzb.x, coords, u, v, w, cc, Ax, Ay, Az, &para_ch[0], job);

                for(int j = 0; j < 8; j++){
                    for(int i = 0; i < 5; i++){
                         stencil_ch[5*j+i] = stencil[8*i+j];//转置
                    }

                    flux_charteric_ptoc_kernel(&stencil_ch[5*j], &para_ch[0]);

                    for(int i = 0; i < 5; i++){
                         stencil[8*i+j] = stencil_ch[5*j+i];//转置
                    }
                }
            }//特征重构
          }

          for(int i = 0; i < 5; i++){

		    flag = OCFD_bound_scheme_kernel_m(&tmp_r, flagxyzb, coords, &stencil[8*i], ia1, ib1, job); 

            if(flag != 0){
                if(Hyscheme_flag == 1){
                     tmp_r = OCFD_UP7_kernel_M(&stencil[8*i]);
                }else if(Hyscheme_flag == 2){
                     tmp_r = OCFD_weno7_kernel_M(&stencil[8*i+1]);
                }else{
                     tmp_r = OCFD_weno5_kernel_M(&stencil[8*i+2]);
                }
            }

            stencil[i] = tmp_r;

		}

        if(Hyscheme_flag != 1) flux_charteric_ctop_kernel(&stencil[0], &para_ch[0]);

        for(int i = 0; i < 5; i++){

            tmp_r = stencil[i];
 
            tmp_l = __shfl_up_double(tmp_r, 1, warpSize);
 
            if(threadIdx.x != 0) put_du_character_m_kernel(flagxyzb, coords, tmp[i], tmp_r, tmp_l, du, i, Ajac, job);
 
        }
	}
}

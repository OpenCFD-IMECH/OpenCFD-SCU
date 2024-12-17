#include "OCFD_ana.h"

#ifdef __cplusplus
extern "C"{
#endif

__global__ void get_inner_kernel(cudaField x1, cudaField x2, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
	
	if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field(x1, x-LAP, y-LAP, z-LAP) = get_Field_LAP(x2, x, y, z);
	}
}

void get_inner(cudaField x1, cudaField x2){
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , nz );
    cudaJobPackage job(dim3(LAP,LAP,LAP) , dim3(nx_lap,ny_lap,nz_lap));
    get_inner_kernel<<<griddim , blockdim>>>(x1, x2, job);
}

void ana_Jac(){
    // check NAN in d u v w T
    // check Negative T 
    int i,j,k,flag = 0;
    unsigned long int offset;

    memcpy_All(pAjac , pAjac_d->ptr , pAjac_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);

    for(k=0;k<nz_2lap;k++){
        for(j=0;j<ny_2lap;j++){
            for(i=0;i<nx_2lap;i++){
                offset = i + nx_2lap*(j + k*ny_2lap);
                if( *(pAjac + offset) < 0 ){
                    printf("\033[31mNegative Jac occured in %d , %d , %d\033[0m\n",
                     i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
                    flag = 1;
                    //goto end_Jac;
                }
            }
        }
    }
    //end_Jac:;

    if(flag == 1) exit(0);
}

__global__ void add_E_kernel(cudaField pE, int SMEMDIM, REAL *g_odata, cudaJobPackage job){
    extern __shared__ REAL shared[];
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int Id = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int warpId  = Id / warpSize;
    unsigned int laneIdx = Id % warpSize;
    REAL grad_f0 = 0.;
    
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        grad_f0 = get_Field(pE, x, y, z);
    }

    grad_f0 = warpReduce(grad_f0);

    if(laneIdx == 0) shared[warpId] = grad_f0;
    __syncthreads();

    grad_f0 = (Id < SMEMDIM)?shared[Id]:0;

    if(warpId == 0) grad_f0 = warpReduce(grad_f0);
    if(Id == 0) g_odata[blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z] = grad_f0;
}

void ana_residual(cudaField PE_d, REAL *E0){

    dim3 size, griddim, blockdim;
    cudaJobPackage job(dim3(0, 0, 0), dim3(nx, ny, nz));
	jobsize(&job, &size);
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    REAL *g_odata;
    REAL *Sum = (REAL *)malloc(sizeof(REAL));

    unsigned int g_odata_size = griddim.x*griddim.y*griddim.z;
    CUDA_LAUNCH(( cudaMalloc((REAL **)&g_odata, g_odata_size*sizeof(REAL)) ));

    int SMEMDIM = blockdim.x*blockdim.y*blockdim.z/64;   //Warpsize is 64
    CUDA_LAUNCH((add_E_kernel<<<griddim, blockdim, SMEMDIM*sizeof(REAL)>>>(PE_d, SMEMDIM, g_odata, job)));

    dim3 blockdim_sum(512);
    dim3 griddim_sum(g_odata_size); 

    do{
        griddim_sum.x = (griddim_sum.x + blockdim_sum.x - 1)/blockdim_sum.x;
        CUDA_LAUNCH(( add_kernel<<<griddim_sum, blockdim_sum, 8*sizeof(REAL)>>>(g_odata, g_odata_size) ));
    } while(griddim_sum.x > 1);

    CUDA_LAUNCH(( cudaMemcpy(Sum, g_odata, sizeof(REAL), cudaMemcpyDeviceToHost) ));
    CUDA_LAUNCH(( cudaFree(g_odata) ));

    MPI_Allreduce(Sum, E0, 1, OCFD_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD);

    *E0 /= NX_GLOBAL * NY_GLOBAL * NZ_GLOBAL;
}

void ana_NAN_and_NT(){
    // check NAN in d u v w T
    // check Negative T

    //if(N_ana < 0 || Istep % Kstep_ana != 0) return;
    int i,j,k;
    unsigned long int offset;

    unsigned int n_NT_limit = 10;
    char has_nan = 0;
    unsigned long int n_NT = 0;


    //memcpy_All(pd , pd_d->ptr , pd_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    //memcpy_All(pu , pu_d->ptr , pu_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    //memcpy_All(pv , pv_d->ptr , pv_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    //memcpy_All(pw , pw_d->ptr , pw_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pT , pT_d->ptr , pT_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);

    if(my_id == 0) printf("It is analyzing NAN......\n");
     
    //for(k=0;k<nz_2lap;k++){
    //    for(j=0;j<ny_2lap;j++){
    //        for(i=0;i<nx_2lap;i++){
    //            offset = i + nx_2lap*(j + k*ny_2lap);
    //            if( isnan( *(pd + offset) ) ){
    //                has_nan = 1;
    //                printf("\033[31mNAN occured in d(%d , %d , %d)\033[0m\non Proc(%d , %d , %d) , global Idx(%d , %d , %d)\n\n",i,j,k,npx,npy,npz , i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
    //                goto end_d;
    //            }
    //        }
    //    }
    //}
    //end_d:;

    //for(k=0;k<nz_2lap;k++){
    //    for(j=0;j<ny_2lap;j++){
    //        for(i=0;i<nx_2lap;i++){
    //            offset = i + nx_2lap*(j + k*ny_2lap);
    //            if( isnan( *(pu + offset) ) ){
    //                has_nan = 1;
    //                printf("\033[31mNAN occured in u(%d , %d , %d)\033[0m\non Proc(%d , %d , %d) , global Idx(%d , %d , %d)\n\n",i,j,k,npx,npy,npz , i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
    //                goto end_u;
    //            }
    //        }
    //    }
    //}
    //end_u:;

    //for(k=0;k<nz_2lap;k++){
    //    for(j=0;j<ny_2lap;j++){
    //        for(i=0;i<nx_2lap;i++){
    //            offset = i + nx_2lap*(j + k*ny_2lap);
    //            if( isnan( *(pv + offset) ) ){
    //                has_nan = 1;
    //                printf("\033[31mNAN occured in v(%d , %d , %d)\033[0m\non Proc(%d , %d , %d) , global Idx(%d , %d , %d)\n\n",i,j,k,npx,npy,npz , i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
    //                goto end_v;
    //            }
    //        }
    //    }
    //}
    //end_v:;

    //for(k=0;k<nz_2lap;k++){
    //    for(j=0;j<ny_2lap;j++){
    //        for(i=0;i<nx_2lap;i++){
    //            offset = i + nx_2lap*(j + k*ny_2lap);
    //            if( isnan( *(pw + offset) ) ){
    //                has_nan = 1;
    //                printf("\033[31mNAN occured in w(%d , %d , %d)\033[0m\non Proc(%d , %d , %d) , global Idx(%d , %d , %d)\n\n",i,j,k,npx,npy,npz , i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
    //                goto end_w;
    //            }
    //        }
    //    }
    //}
    //end_w:;

    for(k=0;k<nz_2lap;k++){
        for(j=0;j<ny_2lap;j++){
            for(i=0;i<nx_2lap;i++){
                offset = i + nx_2lap*(j + k*ny_2lap);
                if( isnan( *(pT + offset) ) ){
                    has_nan = 1;
                    //printf("\033[31mNAN occured in T(%d , %d , %d)\033[0m\non Proc(%d , %d , %d) , global Idx(%d , %d , %d)\n\n",i,j,k,npx,npy,npz , i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
                    printf("\033[31mNAN occured in Global ID(%d , %d , %d)\033[0m\n\n",i_offset[npx]+i-LAP,j_offset[npy]+j-LAP,k_offset[npz]+k-LAP);
                    goto end_T;
                }
            }
        }
    }
    end_T:;

    for(k=0;k<nz_2lap;k++){
        for(j=0;j<ny_2lap;j++){
            for(i=0;i<nx_2lap;i++){
                offset = i + nx_2lap*(j + k*ny_2lap);
                if( *(pT + offset) < 0 ){
                    n_NT++;
                    //printf("\033[31mNegative T occured in T(%d , %d , %d)\033[0m\non Proc(%d , %d , %d) , global Idx(%d , %d , %d)\n\n",i,j,k,npx,npy,npz , i_offset[npx]+i,j_offset[npy]+j,k_offset[npz]+k);
                    printf("\033[31mNegative T occured in Global ID(%d , %d , %d)\033[0m\n\n",i_offset[npx]+i-LAP,j_offset[npy]+j-LAP,k_offset[npz]+k-LAP);
                }
            }
        }
    }
    if(n_NT > n_NT_limit){
        printf("\033[31mNegative T points %ld > %d\033[0m on Proc(%d , %d , %d)\033[0m\n",n_NT , n_NT_limit,npx,npy,npz);
        MPI_Abort(MPI_COMM_WORLD , 1);
    }
    if( has_nan ){
        if(my_id == 0) printf("\033[31mNAN occured , program Abort\033[0m\n");
        MPI_Abort(MPI_COMM_WORLD , 1);
    }

    //cudaStreamDestroy(ana_NT_stream);
}

__global__ void init_time_average_kernel(cudaField d1, cudaField u1, cudaField v1, cudaField w1, cudaField T1, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    //REAL a = get_Field_LAP(d, x, y, z);
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(d1, x, y, z) = 0.;
        get_Field_LAP(u1, x, y, z) = 0.;
        get_Field_LAP(v1, x, y, z) = 0.;
        get_Field_LAP(w1, x, y, z) = 0.;
        get_Field_LAP(T1, x, y, z) = 0.;
    }
}

__global__ void ana_time_average_kernel(cudaField d1, cudaField u1, cudaField v1, cudaField w1, cudaField T1, 
    cudaField d, cudaField u, cudaField v, cudaField w, cudaField T, int Istep, cudaJobPackage job){
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;
    //REAL a = get_Field_LAP(d, x, y, z);
    if(x < job.end.x && y < job.end.y && z < job.end.z){
        get_Field_LAP(d1, x, y, z) = (Istep * get_Field_LAP(d1, x, y, z) + get_Field_LAP(d, x, y, z))/(Istep + 1.);
        get_Field_LAP(u1, x, y, z) = (Istep * get_Field_LAP(u1, x, y, z) + get_Field_LAP(u, x, y, z))/(Istep + 1.);
        get_Field_LAP(v1, x, y, z) = (Istep * get_Field_LAP(v1, x, y, z) + get_Field_LAP(v, x, y, z))/(Istep + 1.);
        get_Field_LAP(w1, x, y, z) = (Istep * get_Field_LAP(w1, x, y, z) + get_Field_LAP(w, x, y, z))/(Istep + 1.);
        get_Field_LAP(T1, x, y, z) = (Istep * get_Field_LAP(T1, x, y, z) + get_Field_LAP(T, x, y, z))/(Istep + 1.);
    }
}

void ana_time_average(){
    if(my_id == 0) printf("It is averaging......\n");
    if(average_IO == 1){
        int tmp_size = (nx + 2 * LAP) * (ny + 2 * LAP) * (nz + 2 * LAP) * sizeof(REAL);
        pdm = (REAL *)malloc_me(tmp_size);
        pum = (REAL *)malloc_me(tmp_size);
        pvm = (REAL *)malloc_me(tmp_size);
        pwm = (REAL *)malloc_me(tmp_size);
        pTm = (REAL *)malloc_me(tmp_size);
    
        new_cudaField(&pdm_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        new_cudaField(&pum_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        new_cudaField(&pvm_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        new_cudaField(&pwm_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        new_cudaField(&pTm_d , nx+2*LAP , ny+2*LAP , nz+2*LAP);

        read_file(average_IO, pdm, pum, pvm, pwm, pTm);
    }

    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , nz );

    cudaJobPackage job(dim3(LAP,LAP,LAP) , dim3(nx_lap,ny_lap,nz_lap));

    CUDA_LAUNCH(( ana_time_average_kernel<<<griddim , blockdim>>>(*pdm_d, *pum_d, *pvm_d, *pwm_d, *pTm_d, 
                                               *pd_d, *pu_d, *pv_d, *pw_d, *pT_d, Istep_average, job) ));

    Istep_average += 1;
    tt_average += dt;

    if(Istep%Kstep_save == 0){
        memcpy_All(pdm , pdm_d->ptr , pdm_d->pitch , D2H , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        memcpy_All(pum , pum_d->ptr , pum_d->pitch , D2H , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        memcpy_All(pvm , pvm_d->ptr , pvm_d->pitch , D2H , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        memcpy_All(pwm , pwm_d->ptr , pwm_d->pitch , D2H , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        memcpy_All(pTm , pTm_d->ptr , pTm_d->pitch , D2H , nx+2*LAP , ny+2*LAP , nz+2*LAP);
        OCFD_save(1, Istep_average, pdm, pum, pvm, pwm, pTm);
    }

    if(tt == end_time){
        free(pdm);
        free(pum);
        free(pvm);
        free(pwm);
        free(pTm);

        delete_cudaField(pdm_d);
        delete_cudaField(pum_d);
        delete_cudaField(pvm_d);
        delete_cudaField(pwm_d);
        delete_cudaField(pTm_d);
    }
}


void init_time_average(){
    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , nz );

    cudaJobPackage job(dim3(0,0,0) , dim3(nx_2lap,ny_2lap,nz_2lap));

    CUDA_LAUNCH(( init_time_average_kernel<<<griddim , blockdim>>>(*pdm_d, *pum_d, *pvm_d, *pwm_d, *pTm_d, job) ));
}

__global__ void get_Q_kernal(
    cudaField ui,
    cudaField us,
    cudaField uk,
    cudaField vi,
    cudaField vs,
    cudaField vk,
    cudaField wi,
    cudaField ws,
    cudaField wk,
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
    cudaField Q,
    cudaJobPackage job){

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    if(x < job.end.x && y < job.end.y && z < job.end.z){
        REAL ux = get_Field(uk, x, y, z)*get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP)+
                  get_Field(ui, x, y, z)*get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP)+
                  get_Field(us, x, y, z)*get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);

        REAL vx = get_Field(vk, x, y, z)*get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP)+
                  get_Field(vi, x, y, z)*get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP)+
                  get_Field(vs, x, y, z)*get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);

        REAL wx = get_Field(wk, x, y, z)*get_Field_LAP(Akx, x+LAP, y+LAP, z+LAP)+
                  get_Field(wi, x, y, z)*get_Field_LAP(Aix, x+LAP, y+LAP, z+LAP)+
                  get_Field(ws, x, y, z)*get_Field_LAP(Asx, x+LAP, y+LAP, z+LAP);

        REAL uy = get_Field(uk, x, y, z)*get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP)+
                  get_Field(ui, x, y, z)*get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP)+
                  get_Field(us, x, y, z)*get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);

        REAL vy = get_Field(vk, x, y, z)*get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP)+
                  get_Field(vi, x, y, z)*get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP)+
                  get_Field(vs, x, y, z)*get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);

        REAL wy = get_Field(wk, x, y, z)*get_Field_LAP(Aky, x+LAP, y+LAP, z+LAP)+
                  get_Field(wi, x, y, z)*get_Field_LAP(Aiy, x+LAP, y+LAP, z+LAP)+
                  get_Field(ws, x, y, z)*get_Field_LAP(Asy, x+LAP, y+LAP, z+LAP);

        REAL uz = get_Field(uk, x, y, z)*get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP)+
                  get_Field(ui, x, y, z)*get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP)+
                  get_Field(us, x, y, z)*get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);

        REAL vz = get_Field(vk, x, y, z)*get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP)+
                  get_Field(vi, x, y, z)*get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP)+
                  get_Field(vs, x, y, z)*get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);

        REAL wz = get_Field(wk, x, y, z)*get_Field_LAP(Akz, x+LAP, y+LAP, z+LAP)+
                  get_Field(wi, x, y, z)*get_Field_LAP(Aiz, x+LAP, y+LAP, z+LAP)+
                  get_Field(ws, x, y, z)*get_Field_LAP(Asz, x+LAP, y+LAP, z+LAP);

        get_Field_LAP(Q, x+LAP, y+LAP, z+LAP) = (ux*vy + ux*wz + vy*wz - uy*vx - uz*wx - vz*wy)*
                                get_Field_LAP(Ajac, x+LAP, y+LAP, z+LAP)*
                                get_Field_LAP(Ajac, x+LAP, y+LAP, z+LAP);
    }
}

void get_Q(){
    cudaField *ui; new_cudaField(&ui, nx, ny, nz);
    cudaField *us; new_cudaField(&us, nx, ny, nz);
    cudaField *uk; new_cudaField(&uk, nx, ny, nz);
    cudaField *vi; new_cudaField(&vi, nx, ny, nz);
    cudaField *vs; new_cudaField(&vs, nx, ny, nz);
    cudaField *vk; new_cudaField(&vk, nx, ny, nz);
    cudaField *wi; new_cudaField(&wi, nx, ny, nz);
    cudaField *ws; new_cudaField(&ws, nx, ny, nz);
    cudaField *wk; new_cudaField(&wk, nx, ny, nz);
    cudaField *Q_d; new_cudaField(&Q_d, nx, ny, nz);

    cudaJobPackage job( dim3(LAP,LAP,LAP) , dim3(nx_lap, ny_lap, nz_lap) );

    OCFD_dx0(*pu_d, *uk, job, BlockDim_X, &Stream[0], D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pv_d, *vk, job, BlockDim_X, &Stream[0], D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pw_d, *wk, job, BlockDim_X, &Stream[0], D0_bound[0], D0_bound[1]);
    OCFD_dy0(*pu_d, *ui, job, BlockDim_Y, &Stream[0], D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pv_d, *vi, job, BlockDim_Y, &Stream[0], D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pw_d, *wi, job, BlockDim_Y, &Stream[0], D0_bound[2], D0_bound[3]);
    OCFD_dz0(*pu_d, *us, job, BlockDim_Z, &Stream[0], D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pv_d, *vs, job, BlockDim_Z, &Stream[0], D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pw_d, *ws, job, BlockDim_Z, &Stream[0], D0_bound[4], D0_bound[5]);

    dim3 griddim , blockdim;
    cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, nz);
    job.setup( dim3(0,0,0) , dim3(nx,ny,nz) );

    CUDA_LAUNCH(( get_Q_kernal<<<griddim, blockdim>>>(*ui,*us,*uk,*vi,*vs,*vk,*wi,*ws,*wk,*pAkx_d,
        *pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,*pAjac_d,*Q_d,job) ));


    memcpy_All(pP, Q_d->ptr, Q_d->pitch, D2H, nx_2lap, ny_2lap, nz_2lap);

    MPI_File tmp_file;
    MPI_File_open(MPI_COMM_WORLD, "Q.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &tmp_file);

    write_3d1(tmp_file, 0, pP);

    MPI_File_close(&tmp_file);

    char filename[100];
    FILE *fp;

    sprintf(filename, "Q%02d%02d%02d.dat", npx, npy, npz);
    fp = fopen(filename, "w");

    fprintf(fp, "variables=x,y,z,Q\n");
    fprintf(fp, "zone i=%d ,j=%d ,k=%d\n", nx, ny, nz);

    for(int k = LAP; k < nz+LAP; k++){
        for(int j = LAP; j < ny+LAP; j++){
            for(int i = LAP; i < nx+LAP; i++){
                fprintf(fp, "%15.6f%15.6f%15.6f%15.6f\n", *(pAxx+i+j*nx_2lap+k*nx_2lap*ny_2lap), *(pAyy+i+j*nx_2lap+k*nx_2lap*ny_2lap), *(pAzz+i+j*nx_2lap+k*nx_2lap*ny_2lap), *(pP+i+j*nx_2lap+k*nx_2lap*ny_2lap));
            }
        }
    }
    
    delete_cudaField(ui);
    delete_cudaField(us);
    delete_cudaField(uk);
    delete_cudaField(vi);
    delete_cudaField(vs);
    delete_cudaField(vk);
    delete_cudaField(wi);
    delete_cudaField(ws);
    delete_cudaField(wk);
    delete_cudaField(Q_d);

    exit(0);
}

void ana_saveplaneXY(int ID){
    int point = ANA_npara[ID][0];
    int bandwidth = ANA_npara[ID][1];

    FILE *fp; 
    char fp_name[120];

    memcpy_All(pd , pd_d->ptr , pd_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pu , pu_d->ptr , pu_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pv , pv_d->ptr , pv_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pw , pw_d->ptr , pw_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pT , pT_d->ptr , pT_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);

    for(int i = 0; i < point; i++){

        if(my_id == 0){
            printf("Save data ...., %d, %lf, %d\n", Istep, tt, i);
            sprintf(fp_name, "Savedata-XY%03d.dat", i);
            fp = fopen(fp_name, "a");

            int bytes = sizeof(REAL) + sizeof(int);

            fwrite(&bytes, sizeof(int), 1, fp);
            fwrite(&Istep, sizeof(int), 1, fp);
            fwrite(&tt, sizeof(REAL), 1, fp);
            fwrite(&bytes, sizeof(int), 1, fp);
        }

        for(int j = ANA_npara[ID][i+2]; j <= ANA_npara[ID][i+2]+bandwidth-1; j++){
            write_2d_XYa(fp, j, pd);
            write_2d_XYa(fp, j, pu);
            write_2d_XYa(fp, j, pv);
            write_2d_XYa(fp, j, pw);
            write_2d_XYa(fp, j, pT);
        }

        if(my_id == 0) fclose(fp);
    }
}

void ana_saveplaneYZ(int ID){
    int point = ANA_npara[ID][0];
    int bandwidth = ANA_npara[ID][1];

    FILE *fp; 
    char fp_name[120];

    memcpy_All(pd , pd_d->ptr , pd_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pu , pu_d->ptr , pu_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pv , pv_d->ptr , pv_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pw , pw_d->ptr , pw_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pT , pT_d->ptr , pT_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);

    for(int i = 0; i < point; i++){

        if(my_id == 0){
            printf("Save data ...., %d, %lf, %d\n", Istep, tt, i);
            sprintf(fp_name, "Savedata-YZ%03d.dat", i);
            fp = fopen(fp_name, "a");

            //fprintf(fp, "%d%lf\n", Istep, tt);
            int bytes = sizeof(REAL) + sizeof(int);

            fwrite(&bytes, sizeof(int), 1, fp);
            fwrite(&Istep, sizeof(int), 1, fp);
            fwrite(&tt, sizeof(REAL), 1, fp);
            fwrite(&bytes, sizeof(int), 1, fp);
        }

        for(int j = ANA_npara[ID][i+2]; j <= ANA_npara[ID][i+2]+bandwidth-1; j++){
            write_2d_YZa(fp, j, pd);
            write_2d_YZa(fp, j, pu);
            write_2d_YZa(fp, j, pv);
            write_2d_YZa(fp, j, pw);
            write_2d_YZa(fp, j, pT);
        }

        if(my_id == 0) fclose(fp);
    }
}

void ana_saveplaneXZ(int ID){
    int point = ANA_npara[ID][0];
    int bandwidth = ANA_npara[ID][1]; 

    FILE *fp; 
    char fp_name[120];

    memcpy_All(pd , pd_d->ptr , pd_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pu , pu_d->ptr , pu_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pv , pv_d->ptr , pv_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pw , pw_d->ptr , pw_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
    memcpy_All(pT , pT_d->ptr , pT_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);

    for(int i = 0; i < point; i++){

        if(my_id == 0){
            printf("Save data ...., %d, %lf, %d\n", Istep, tt, i);
            sprintf(fp_name, "Savedata-XZ%03d.dat", i);
            fp = fopen(fp_name, "a");

            //fprintf(fp, "%d%lf\n", Istep, tt);
            int bytes = sizeof(REAL) + sizeof(int);

            fwrite(&bytes, sizeof(int), 1, fp);
            fwrite(&Istep, sizeof(int), 1, fp);
            fwrite(&tt, sizeof(REAL), 1, fp);
            fwrite(&bytes, sizeof(int), 1, fp);
        }

        for(int j = ANA_npara[ID][i+2]; j <= ANA_npara[ID][i+2]+bandwidth-1; j++){
            write_2d_XZa(fp, j, pd);
            write_2d_XZa(fp, j, pu);
            write_2d_XZa(fp, j, pv);
            write_2d_XZa(fp, j, pw);
            write_2d_XZa(fp, j, pT);
        }

        if(my_id == 0) fclose(fp);
    }
}

void OCFD_ana(int style, int ID){
    switch(style){
        case 100:
        ana_NAN_and_NT();
        break;

        case 101:
        ana_time_average();
        break;

        case 102:
        HybridAuto_scheme_IO();
        break;

        case 103:
        get_Q();
        break;

        case 104:
        ana_saveplaneXY(ID);
        break;

        case 105:
        ana_saveplaneYZ(ID);
        break;

        case 106:
        ana_saveplaneXZ(ID);
        break;

        case 107:
        if(IFLAG_HybridAuto == 1) HybridAuto_scheme_Proportion();
        break;
    }
}

#ifdef __cplusplus
}
#endif

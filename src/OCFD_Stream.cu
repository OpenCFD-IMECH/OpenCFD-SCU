#include <stdio.h>
#include <stdlib.h>

#include "OCFD_Stream.h"
#include "OCFD_split.h"
#include "OCFD_NS_Jacobian3d.h"
#include "parameters.h"
#include "OCFD_mpi_dev.h"
#include "parameters_d.h"
#include "commen_kernel.h"
#include "OCFD_Schemes_hybrid_auto.h"
#ifdef __cplusplus
extern "C" {
#endif

//static cudaStream_t Stream[15];

void opencfd_mem_init_Stream(){
    for (int i = 0; i < 4; i++) cudaStreamCreate(&Stream[i]);
    for (int i = 0; i < 4; i++) cudaEventCreate(&Event[i]);
}

void opencfd_mem_finalize_Stream(){
    for (int i = 0; i < 4; ++i) cudaStreamDestroy(Stream[i]);
    for (int i = 0; i < 4; ++i) cudaEventDestroy(Event[i]);
}

void du_comput(int KRK){
	//pthread_create(&thread_handles[0], NULL, du_invis_Jacobian3d_inner, NULL);
	//pthread_create(&thread_handles[1], NULL, du_vis_Jacobian3d_outer, NULL);

	//for(int thread = 0; thread < 2; thread++)
	//	pthread_join(thread_handles[thread], NULL);
	if(IFLAG_HybridAuto == 1 && KRK == 1) Set_Scheme_HybridAuto(&Stream[0]);

	cuda_mem_value_init_warp(0.0 ,pdu_d->ptr, pdu_d->pitch, nx, ny, nz*5);

	switch(Stream_MODE){
        case 0://Non-stream
	    du_invis_Jacobian3d(NULL);
	    du_vis_Jacobian3d(NULL);
        break;

        case 1://launch: first invis, then vis
        //du_invis_Jacobian3d_all(NULL);
	    //du_vis_Jacobian3d_all(NULL);
        du_Jacobian3d_all(NULL);
		break;
		
		default: 
		if(my_id == 0) printf("\033[31mWrong Stream Mode! Please choose 0 or 1, 0:non stream; 1:stream\033[0m\n");
    }

}

/*

void *du_Jacobian3d_all(void* pthread_id){
    cudaJobPackage job(dim3(2*LAP, 2*LAP, 2*LAP), dim3(nx, ny, nz));
//cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, nz_lap));

	du_invis_Jacobian3d_init(job, &Stream[0]);//内区声速计算

	job.setup(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, &Stream[0]);
    du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, &Stream[0]);//内区无粘项计算
    du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, &Stream[0]);
    du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, &Stream[0]);

    du_invis_Jacobian3d_outer_exchange(&Stream[2]);//交换原始变量
    cudaDeviceSynchronize();

    du_invis_Jacobian3d_outer_x(&Stream[1]);//外区计算
    du_invis_Jacobian3d_outer_y(&Stream[1]);
    du_invis_Jacobian3d_outer_z(&Stream[1]);

    cudaDeviceSynchronize();

    du_viscous_Jacobian3d_init(&Stream[2]);//开始算粘性项全体导数
    cudaDeviceSynchronize();

    du_viscous_Jacobian3d_x_init(&Stream[2]);//粘性项计算
    du_vis_Jacobian3d_inner_x(&Stream[2]);//内区开始计算
    cudaDeviceSynchronize();

    du_vis_Jacobian3d_outer_x(&Stream[3]);//外区x计算
    cudaDeviceSynchronize();

    du_viscous_Jacobian3d_y_init(&Stream[2]);
    du_vis_Jacobian3d_inner_y(&Stream[2]);//内区开始计算
    cudaDeviceSynchronize();

    du_vis_Jacobian3d_outer_y(&Stream[3]);//外区x计算
    cudaDeviceSynchronize();

    du_viscous_Jacobian3d_z_init(&Stream[2]);
    du_vis_Jacobian3d_inner_z(&Stream[2]);//内区开始计算
    cudaDeviceSynchronize();

    du_vis_Jacobian3d_outer_z(&Stream[3]);//外区x计算

	return NULL;
}
*/


void *du_Jacobian3d_all(void* pthread_id){
    cudaJobPackage job(dim3(2*LAP, 2*LAP, 2*LAP), dim3(nx, ny, nz));
	du_invis_Jacobian3d_init(job, &Stream[0]);//内区声速计算

	job.setup(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, &Stream[0]);
    du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, &Stream[0]);//内区无粘项计算
    du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, &Stream[0]);
    du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, &Stream[0]);

    du_invis_Jacobian3d_outer_exchange(&Stream[1]);//交换原始变量

    cudaEventRecord(Event[1], Stream[1]);//记录数据交换情况
    cudaStreamWaitEvent(Stream[2], Event[1], 0);//外区等待粘性计算完

    du_invis_Jacobian3d_outer_x(&Stream[1]);//外区计算

    du_viscous_Jacobian3d_init(&Stream[2]);//开始算粘性项全体导数
    du_viscous_Jacobian3d_x_init(&Stream[2]);//粘性项计算
    cudaEventRecord(Event[2], Stream[2]);//记录粘性计算
    du_vis_Jacobian3d_inner_x(&Stream[2]);//内区开始计算

    cudaStreamWaitEvent(Stream[3], Event[2], 0);//外区等待粘性计算完
    du_vis_Jacobian3d_outer_x(&Stream[3]);//外区x计算
    cudaEventRecord(Event[3], Stream[3]);

    du_invis_Jacobian3d_outer_y(&Stream[1]);

    cudaStreamWaitEvent(Stream[2], Event[3], 0);
    du_viscous_Jacobian3d_y_init(&Stream[2]);
    cudaEventRecord(Event[2], Stream[2]);//记录粘性计算
    du_vis_Jacobian3d_inner_y(&Stream[2]);//内区开始计算

    cudaStreamWaitEvent(Stream[3], Event[2], 0);//外区等待粘性计算完 
    du_vis_Jacobian3d_outer_y(&Stream[3]);//外区x计算
    cudaEventRecord(Event[3], Stream[3]);

    du_invis_Jacobian3d_outer_z(&Stream[1]);

    cudaStreamWaitEvent(Stream[2], Event[3], 0);
    du_viscous_Jacobian3d_z_init(&Stream[2]);
    cudaEventRecord(Event[2], Stream[2]);//记录粘性计算
    du_vis_Jacobian3d_inner_z(&Stream[2]);//内区开始计算

    cudaStreamWaitEvent(Stream[3], Event[2], 0);
    du_vis_Jacobian3d_outer_z(&Stream[3]);//外区x计算

	return NULL;
}

/*void* du_invis_Jacobian3d_all(void* pthread_id){

	cudaJobPackage job(dim3(2*LAP, 2*LAP, 2*LAP), dim3(nx, ny, nz));
	du_invis_Jacobian3d_init(job, &Stream[0]);

	job.setup(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
    //direction X ------------------------------
	
	du_invis_Jacobian3d_x(job, pfp_d, pfm_d, &Stream[0]);
	du_invis_Jacobian3d_outer_x(&Stream[1]);

    //direction Y ------------------------------

	cudaEventRecord(Event[0], Stream[0]);
	cudaEventRecord(Event[1], Stream[1]);
	cudaStreamWaitEvent(Stream[0], Event[1], 0);
	du_invis_Jacobian3d_y(job, pfp_d, pfm_d, &Stream[0]);
	du_invis_Jacobian3d_outer_y(&Stream[1], &Event[0]);

    //direction Z ------------------------------

	cudaEventRecord(Event[0], Stream[0]);
	cudaEventRecord(Event[1], Stream[1]);
	cudaStreamWaitEvent(Stream[0], Event[1], 0);
	du_invis_Jacobian3d_z(job, pfp_d, pfm_d, &Stream[0]);
	du_invis_Jacobian3d_outer_z(&Stream[1], &Event[0]);
	cudaEventRecord(Event[1], Stream[1]);

	return NULL;
}*/

//void* du_vis_Jacobian3d_all(void* pthread_id){
//
//	cudaStreamWaitEvent(Stream[2], Event[1], 0);
//	du_viscous_Jacobian3d_init(&Stream[2]);
//
//    //direction X ------------------------------
//	
//	du_viscous_Jacobian3d_x_init(&Stream[2]);
//	cudaEventRecord(Event[2], Stream[2]);
//	du_vis_Jacobian3d_inner_x(&Stream[2]);
//	cudaStreamWaitEvent(Stream[1], Event[2], 0);
//	du_vis_Jacobian3d_outer_x(&Stream[1]);
//
//    //direction Y ------------------------------
//
//	cudaEventRecord(Event[2], Stream[1]);
//	cudaStreamWaitEvent(Stream[2], Event[2], 0);
//	du_viscous_Jacobian3d_y_init(&Stream[2]);
//	cudaEventRecord(Event[2], Stream[2]);
//	du_vis_Jacobian3d_inner_y(&Stream[2]);
//	cudaStreamWaitEvent(Stream[1], Event[2], 0);
//	du_vis_Jacobian3d_outer_y(&Stream[1]);
//
//    //direction X ------------------------------
//	
//	cudaEventRecord(Event[2], Stream[1]);
//	cudaStreamWaitEvent(Stream[2], Event[2], 0);
//	du_viscous_Jacobian3d_z_init(&Stream[2]);
//	cudaEventRecord(Event[2], Stream[2]);
//	du_vis_Jacobian3d_inner_z(&Stream[2]);
//	cudaStreamWaitEvent(Stream[1], Event[2], 0);
//	du_vis_Jacobian3d_outer_z(&Stream[1]);
//	
//
//	return NULL;
//}
void* du_vis_Jacobian3d_all(void* pthread_id){

	du_viscous_Jacobian3d_init(&Stream[2]);

    //direction X ------------------------------
	
	du_viscous_Jacobian3d_x_init(&Stream[2]);
	cudaEventRecord(Event[2], Stream[2]);
	du_vis_Jacobian3d_inner_x(&Stream[2]);
	cudaStreamWaitEvent(Stream[3], Event[1], 0);
	cudaStreamWaitEvent(Stream[3], Event[2], 0);
	du_vis_Jacobian3d_outer_x(&Stream[3]);

    //direction Y ------------------------------

	cudaEventRecord(Event[2], Stream[3]);
	cudaStreamWaitEvent(Stream[2], Event[2], 0);
	du_viscous_Jacobian3d_y_init(&Stream[2]);
	cudaEventRecord(Event[2], Stream[2]);
	du_vis_Jacobian3d_inner_y(&Stream[2]);
	cudaStreamWaitEvent(Stream[3], Event[2], 0);
	du_vis_Jacobian3d_outer_y(&Stream[3]);

    //direction Z ------------------------------
	
	cudaEventRecord(Event[2], Stream[3]);
	cudaStreamWaitEvent(Stream[2], Event[2], 0);
	du_viscous_Jacobian3d_z_init(&Stream[2]);
	cudaEventRecord(Event[2], Stream[2]);
	du_vis_Jacobian3d_inner_z(&Stream[2]);
	cudaStreamWaitEvent(Stream[3], Event[2], 0);
	du_vis_Jacobian3d_outer_z(&Stream[3]);
	

	return NULL;
}


void* du_vis_Jacobian3d_inner_x(cudaStream_t *stream){

	cudaJobPackage job(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_x_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_inner_y(cudaStream_t *stream){

	cudaJobPackage job(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_y_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_inner_z(cudaStream_t *stream){

	cudaJobPackage job(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_z_final(job, stream);

	return NULL;
}


void* du_invis_Jacobian3d(void* pthread_id){

	exchange_boundary_xyz_packed_dev(pd , pd_d);
	exchange_boundary_xyz_packed_dev(pu , pu_d);
	exchange_boundary_xyz_packed_dev(pv , pv_d); 
	exchange_boundary_xyz_packed_dev(pw , pw_d);
	exchange_boundary_xyz_packed_dev(pT , pT_d);

	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, nz_lap));
	du_invis_Jacobian3d_init(job, &Stream[0]);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, &Stream[0]);
	du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, &Stream[0]);
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, &Stream[0]);
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, &Stream[0]);

	return NULL;
}

void* du_vis_Jacobian3d(void* pthread_id){

	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, nz_lap));
	du_viscous_Jacobian3d_init(&Stream[0]);
	
	du_viscous_Jacobian3d_x_init(&Stream[0]);
	exchange_boundary_x_packed_dev(pEv1 , pEv1_d , Iperiodic[0]);
    exchange_boundary_x_packed_dev(pEv2 , pEv2_d , Iperiodic[0]);
    exchange_boundary_x_packed_dev(pEv3 , pEv3_d , Iperiodic[0]);
    exchange_boundary_x_packed_dev(pEv4 , pEv4_d , Iperiodic[0]);
	du_viscous_Jacobian3d_x_final(job, &Stream[0]);

	du_viscous_Jacobian3d_y_init(&Stream[0]);
	exchange_boundary_y_packed_dev(pEv1 , pEv1_d , Iperiodic[1]);
    exchange_boundary_y_packed_dev(pEv2 , pEv2_d , Iperiodic[1]);
    exchange_boundary_y_packed_dev(pEv3 , pEv3_d , Iperiodic[1]);
	exchange_boundary_y_packed_dev(pEv4 , pEv4_d , Iperiodic[1]);
	boundary_symmetry_pole_vis_y(&Stream[0]);
	du_viscous_Jacobian3d_y_final(job, &Stream[0]);

	du_viscous_Jacobian3d_z_init(&Stream[0]);
	exchange_boundary_z_packed_dev(pEv1 , pEv1_d ,Iperiodic[2]);
    exchange_boundary_z_packed_dev(pEv2 , pEv2_d ,Iperiodic[2]);
    exchange_boundary_z_packed_dev(pEv3 , pEv3_d ,Iperiodic[2]);
    exchange_boundary_z_packed_dev(pEv4 , pEv4_d ,Iperiodic[2]);
	du_viscous_Jacobian3d_z_final(job, &Stream[0]);

	return NULL;
}

void* du_invis_Jacobian3d_outer_init_x(cudaStream_t *stream){
//-------------x outer p init----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_invis_Jacobian3d_init(job, stream);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, stream);
//-------------x outer m init----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_invis_Jacobian3d_init(job, stream);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, stream);
	
	return NULL;
}


void* du_invis_Jacobian3d_outer_x_x(cudaStream_t *stream){
//-------------x outer p x----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, stream);
//-------------x outer m x----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_y_x(cudaStream_t *stream){
//-------------x outer p y----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, stream);
//-------------x outer m y----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_z_x(cudaStream_t *stream){
//-------------x outer p z----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, stream);
//-------------x outer m z----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_init_y(cudaStream_t *stream){
//-------------y outer p init----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_invis_Jacobian3d_init(job, stream);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, stream);

//-------------y outer m init----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_invis_Jacobian3d_init(job, stream);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, stream);
		
	return NULL;
}
	

void* du_invis_Jacobian3d_outer_x_y(cudaStream_t *stream){
//-------------y outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, stream);
//-------------y outer m x----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_y_y(cudaStream_t *stream){
//-------------y outer p y----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, stream);
//-------------y outer m y----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_z_y(cudaStream_t *stream){
//-------------y outer p z----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, stream);
//-------------y outer m z----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_init_z(cudaStream_t *stream){
//-------------z outer p init----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_invis_Jacobian3d_init(job, stream);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, stream);
//-------------z outer m init----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_invis_Jacobian3d_init(job, stream);
    Stager_Warming(job, pfp_x_d, pfm_x_d, pfp_y_d, pfm_y_d, pfp_z_d, pfm_z_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_x_z(cudaStream_t *stream){
//-------------z outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, stream);
//-------------z outer m x----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
    du_invis_Jacobian3d_x(job, pfp_x_d, pfm_x_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_y_z(cudaStream_t *stream){
//-------------z outer p----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, stream);
//-------------z outer m----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_invis_Jacobian3d_y(job, pfp_y_d, pfm_y_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_z_z(cudaStream_t *stream){
//-------------z outer p----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, stream);
//-------------z outer m----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_invis_Jacobian3d_z(job, pfp_z_d, pfm_z_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_exchange(cudaStream_t *stream){

	exchange_boundary_xyz_Async_packed_dev(pd , pd_d , stream);
	exchange_boundary_xyz_Async_packed_dev(pu , pu_d , stream);
	exchange_boundary_xyz_Async_packed_dev(pv , pv_d , stream); 
	exchange_boundary_xyz_Async_packed_dev(pw , pw_d , stream);
	exchange_boundary_xyz_Async_packed_dev(pT , pT_d , stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_x(cudaStream_t *stream){

	du_invis_Jacobian3d_outer_init_x(stream);
	du_invis_Jacobian3d_outer_x_x(stream);
	du_invis_Jacobian3d_outer_y_x(stream);
	du_invis_Jacobian3d_outer_z_x(stream);
	
	return NULL;
}


void* du_invis_Jacobian3d_outer_y(cudaStream_t *stream){
	
	du_invis_Jacobian3d_outer_init_y(stream);
	du_invis_Jacobian3d_outer_x_y(stream);
	du_invis_Jacobian3d_outer_y_y(stream);
	du_invis_Jacobian3d_outer_z_y(stream);
	
	return NULL;
}


void* du_invis_Jacobian3d_outer_z(cudaStream_t *stream){

	du_invis_Jacobian3d_outer_init_z(stream);
	du_invis_Jacobian3d_outer_x_z(stream);
	du_invis_Jacobian3d_outer_y_z(stream);
	du_invis_Jacobian3d_outer_z_z(stream);
	
	return NULL;
}


void* du_vis_Jacobian3d_outer_x_x(cudaStream_t *stream){

//-------------x outer p x----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_x_final(job, stream);
//-------------x outer m x----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_viscous_Jacobian3d_x_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_y_x(cudaStream_t *stream){
//-------------x outer p y----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_y_final(job, stream);
//-------------x outer m y----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_viscous_Jacobian3d_y_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_z_x(cudaStream_t *stream){
//-------------x outer p z----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_z_final(job, stream);
//-------------x outer m z----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_viscous_Jacobian3d_z_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_x_y(cudaStream_t *stream){
//-------------y outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_viscous_Jacobian3d_x_final(job, stream);
//-------------y outer m x----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_viscous_Jacobian3d_x_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_y_y(cudaStream_t *stream){
//-------------y outer p y----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_viscous_Jacobian3d_y_final(job, stream);
//-------------y outer m y----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_viscous_Jacobian3d_y_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_z_y(cudaStream_t *stream){
//-------------y outer p z----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_viscous_Jacobian3d_z_final(job, stream);
//-------------y outer m z----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_viscous_Jacobian3d_z_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_x_z(cudaStream_t *stream){
//-------------z outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_viscous_Jacobian3d_x_final(job, stream);
//-------------z outer m x----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_viscous_Jacobian3d_x_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_outer_y_z(cudaStream_t *stream){
//-------------z outer p y----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_viscous_Jacobian3d_y_final(job, stream);
//-------------z outer m y----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_viscous_Jacobian3d_y_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_outer_z_z(cudaStream_t *stream){
//-------------z outer p z----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_viscous_Jacobian3d_z_final(job, stream);
//-------------z outer m z----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_viscous_Jacobian3d_z_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_outer_x(cudaStream_t *stream){

	exchange_boundary_x_Async_packed_dev(pEv1 , pEv1_d , Iperiodic[0], stream);
    exchange_boundary_x_Async_packed_dev(pEv2 , pEv2_d , Iperiodic[0], stream);
    exchange_boundary_x_Async_packed_dev(pEv3 , pEv3_d , Iperiodic[0], stream);
    exchange_boundary_x_Async_packed_dev(pEv4 , pEv4_d , Iperiodic[0], stream);

	du_vis_Jacobian3d_outer_x_x(stream);
	du_vis_Jacobian3d_outer_x_y(stream);
	du_vis_Jacobian3d_outer_x_z(stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_y(cudaStream_t *stream){

	exchange_boundary_y_Async_packed_dev(pEv1 , pEv1_d , Iperiodic[1], stream);
    exchange_boundary_y_Async_packed_dev(pEv2 , pEv2_d , Iperiodic[1], stream);
    exchange_boundary_y_Async_packed_dev(pEv3 , pEv3_d , Iperiodic[1], stream);
	exchange_boundary_y_Async_packed_dev(pEv4 , pEv4_d , Iperiodic[1], stream);
	
	boundary_symmetry_pole_vis_y(stream);

	du_vis_Jacobian3d_outer_y_x(stream);
	du_vis_Jacobian3d_outer_y_y(stream);
	du_vis_Jacobian3d_outer_y_z(stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_z(cudaStream_t *stream){

	exchange_boundary_z_Async_packed_dev(pEv1 , pEv1_d , Iperiodic[2], stream);
    exchange_boundary_z_Async_packed_dev(pEv2 , pEv2_d , Iperiodic[2], stream);
    exchange_boundary_z_Async_packed_dev(pEv3 , pEv3_d , Iperiodic[2], stream);
    exchange_boundary_z_Async_packed_dev(pEv4 , pEv4_d , Iperiodic[2], stream);

	du_vis_Jacobian3d_outer_z_x(stream);
	du_vis_Jacobian3d_outer_z_y(stream);
	du_vis_Jacobian3d_outer_z_z(stream);

	return NULL;
}

#ifdef __cplusplus
}
#endif

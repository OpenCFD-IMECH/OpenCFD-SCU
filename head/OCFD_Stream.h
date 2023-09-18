#ifndef __OCFD_STREAM_H
#define __OCFD_STREAM_H
#include "cuda.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C"{
#endif

void du_comput(int KRK);

void opencfd_mem_init_Stream();
void opencfd_mem_finalize_Stream();

void* du_Jacobian3d_all(void* pthread_id);

void* du_invis_Jacobian3d(void* pthread_id);
void* du_invis_Jacobian3d_all(void* pthread_id);

void* du_invis_Jacobian3d_outer_exchange(cudaStream_t *stream);
void* du_invis_Jacobian3d_outer_x(cudaStream_t *stream);
void* du_invis_Jacobian3d_outer_y(cudaStream_t *stream);
void* du_invis_Jacobian3d_outer_z(cudaStream_t *stream);

void* du_vis_Jacobian3d(void* pthread_id);
void* du_vis_Jacobian3d_all(void* pthread_id);

void* du_vis_Jacobian3d_inner_x(cudaStream_t *stream);
void* du_vis_Jacobian3d_inner_y(cudaStream_t *stream);
void* du_vis_Jacobian3d_inner_z(cudaStream_t *stream);

void* du_vis_Jacobian3d_outer_x(cudaStream_t *stream);
void* du_vis_Jacobian3d_outer_y(cudaStream_t *stream);
void* du_vis_Jacobian3d_outer_z(cudaStream_t *stream);



#ifdef __cplusplus
}
#endif
#endif


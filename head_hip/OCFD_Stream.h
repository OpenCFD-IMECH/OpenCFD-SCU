#ifndef __OCFD_STREAM_H
#define __OCFD_STREAM_H
#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"

#ifdef __cplusplus
extern "C"{
#endif

void du_comput(int KRK);

void opencfd_mem_init_Stream();
void opencfd_mem_finalize_Stream();

void* du_Jacobian3d_all(void* pthread_id);

void* du_invis_Jacobian3d(void* pthread_id);
void* du_invis_Jacobian3d_all(void* pthread_id);

void* du_invis_Jacobian3d_outer_exchange(hipStream_t *stream);
void* du_invis_Jacobian3d_outer_x(hipStream_t *stream);
void* du_invis_Jacobian3d_outer_y(hipStream_t *stream);
void* du_invis_Jacobian3d_outer_z(hipStream_t *stream);

void* du_vis_Jacobian3d(void* pthread_id);
void* du_vis_Jacobian3d_all(void* pthread_id);

void* du_vis_Jacobian3d_inner_x(hipStream_t *stream);
void* du_vis_Jacobian3d_inner_y(hipStream_t *stream);
void* du_vis_Jacobian3d_inner_z(hipStream_t *stream);

void* du_vis_Jacobian3d_outer_x(hipStream_t *stream);
void* du_vis_Jacobian3d_outer_y(hipStream_t *stream);
void* du_vis_Jacobian3d_outer_z(hipStream_t *stream);



#ifdef __cplusplus
}
#endif
#endif


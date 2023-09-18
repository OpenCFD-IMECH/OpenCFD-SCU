#ifndef __OCFD_INIT_H
#define __OCFD_INIT_H
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif

void init();

void opencfd_mem_init_all();
void opencfd_mem_finalize_all();

void opencfd_mem_init();
void opencfd_mem_finalize();
void opencfd_para_init();

void opencfd_mem_init_boundary();
void opencfd_mem_finalize_boundary();

void opencfd_mem_init_dev();
void opencfd_mem_finalize_dev();
void opencfd_para_init_dev();

#ifdef __cplusplus
}
#endif
#endif

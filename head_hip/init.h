#ifndef __INIT_H
#define __INIT_H
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif

#ifdef __HIPCC__
void opencfd_para_init_dev();
#endif
#ifdef __cplusplus
}
#endif
#endif
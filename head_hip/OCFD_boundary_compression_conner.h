#ifndef __OCFD_BOUNDARY_COMPRESSION_H
#define __OCFD_BOUNDARY_COMPRESSION_H
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif

void get_ht_multifrequancy(REAL HT, REAL TT, int MT_MAX, REAL beta);
void bc_user_Compression_conner();

#ifdef __cplusplus
}
#endif
#endif
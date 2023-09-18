#ifndef __OCFD_IO_MPI_H
#define __OCFD_IO_MPI_H
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif
void write_2d_XY(FILE * file, int ka, int size_x, int size_y, int lap, int *pU, REAL *pU1);
void write_2d_XYa(FILE * file, int ka, REAL *pU);
void write_2d_YZa(FILE * file, int ia, REAL *pU);
void write_2d_XZa(FILE * file, int ja, REAL *pU);
void write_points(FILE * file, REAL * pU, int mpoints, int *ia, int *ja, int *ka);
void read_3d(MPI_File file, MPI_Offset offset, REAL *pU);
void write_3d(MPI_File file, MPI_Offset offset, REAL *pU);
void write_blockdata(FILE * file, REAL * pU, int ib, int ie, int jb, int je, int kb, int ke);

#ifdef __cplusplus
}
#endif
#endif

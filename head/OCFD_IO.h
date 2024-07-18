#ifndef __OCFD_IO_H
#define __OCFD_IO_H
#include "stdio.h"
#include "parameters.h"

#ifdef __cplusplus
extern "C"{
#endif

void read_file(int Iflag_av , REAL * pd , REAL * pu , REAL * pv , REAL * pw , REAL * pT);
void OCFD_save(int Iflag_av, int Istep_name , REAL * pd , REAL * pu , REAL * pv , REAL * pw , REAL * pT);
void OCFD_save_final(int Iflag_av, int Istep_name , REAL * pd , REAL * pu , REAL * pv , REAL * pw , REAL * pT, REAL *O2, REAL *N2);
void write_3d1(MPI_File  pfile, MPI_Offset offset, REAL * pU);
void read_3d1(MPI_File pfile, MPI_Offset offset, REAL * pU);

#ifdef __cplusplus
}
#endif
#endif

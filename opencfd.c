//----------------------------------------------------------------------------------------------------------------------------------------   
// OpenCFD-SC  ,  3-D compressible Navier-Stokes Finite difference Solver 
// Copyright by LI Xinliang, LHD, Institute of Mechanics, CAS, Email: lixl@imech.ac.cn
//  
// The default code is double precision computation
// If you want to use SINGLE PRECISION computation, you can change   "OCFD_REAL_KIND=8"  to "OCFD_REAL_KIND=4" ,
// and  "OCFD_DATA_TYPE=MPI_DOUBLE_PRECISION" to "OCFD_DATA_TYPE=MPI_REAL" in the file OpenCFD.h 
//---------------------------------------------------------------------------------------------------------------------------------------------- 
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#include "utility.h"
#include "parameters.h"

#include "OCFD_NS_Solver.h"
#include "OCFD_mpi.h"
#include "OCFD_init.h"
#include "cuda_commen.h"
#include "OCFD_mpi_dev.h"
#include "OCFD_filtering.h"

#include "opencc.h"

#ifdef PROFILING
#include <cuda_profiler_api.h>
#endif

#ifdef __cplusplus
extern "C"{
#endif

int main(int argc, char *argv[]){
    mpi_init(&argc , &argv);

    read_parameters();

    opencfd_mem_init_mpi();  

    part();

    set_para_filtering();
    
    opencfd_mem_init_all();

    cuda_commen_init();

    init();

    char *fname = "./chem_h2.yaml";
    int size = 5;
    double T[4] = {1,2,3,4};  
    double P[4] = {1,2,3,4};  
    double Y[4] = {1,2,3,4}; 

    opencc_ode_init(fname, size, T, P, Y);
#ifdef PROFILING
cudaProfilerStart();
#endif
    NS_solver_real();
#ifdef PROFILING
cudaProfilerStop();
#endif

    opencfd_mem_finalize_all();

    mpi_finalize();
    
    return 0;
}

#ifdef __cplusplus
}
#endif


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


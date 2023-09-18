#ifndef __NS_SOLVER_C
#define __NS_SOLVER_C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <math.h>

#include "mpi.h"
#include "parameters.h"
#include "utility.h"
#include "OCFD_NS_Solver.h"
#include "OCFD_time.h"
#include "OCFD_mpi.h"
#include "OCFD_boundary.h"
#include "OCFD_IO.h"
#include "OCFD_init.h"
#include "OCFD_Stream.h"
#include "OCFD_filtering.h"
#include "OCFD_ana.h"

#include "OCFD_mpi_dev.h"
#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "commen_kernel.h"

#ifdef __cplusplus
extern "C"{
#endif

void NS_solver_real()
{

    //  initial of Amu, Amu_t ---
    //  Amu=0.d0;
    // ----------------initial---------------------------------------------------------

    exchange_boundary_xyz_packed_dev(pd , pd_d);
    exchange_boundary_xyz_packed_dev(pu , pu_d);
    exchange_boundary_xyz_packed_dev(pv , pv_d);
    exchange_boundary_xyz_packed_dev(pw , pw_d);
    exchange_boundary_xyz_packed_dev(pT , pT_d);

    OCFD_bc();
    
    get_Amu();

    if (my_id == 0)
        printf("init ok\n\n");

    REAL wstart0 , wstart , wend;
    wstart0 = MPI_Wtime();
    wstart = wstart0;
    // -----------------------------------------------------------------------
    do
    {
        {
            REAL * tmp = pf;
            pf = pfn;
            pfn = tmp;

            tmp = pf_d->ptr;
            pf_d->ptr = pfn_d->ptr;
            pfn_d->ptr = tmp;
        }

        // 3-step Runge-Kutta
        for (int KRK = 1; KRK <= 3; KRK++)
        {
            
            du_comput(KRK);
            
            OCFD_time_advance(KRK);
            
            get_duvwT();

            OCFD_bc();

            get_Amu();

        }

        Istep++;
        tt += dt;
        
        // ---Filtering -------------------------------
        filtering(pf, pf_lap , pP);


        //modify_NT();
        for(int i = 0; i < N_ana; i++){
            if(Istep % Kstep_ana[i] == 0) OCFD_ana(K_ana[i], i);
        } 


        if(Istep % Kstep_show == 0){
            MPI_Barrier(MPI_COMM_WORLD);
            wend = MPI_Wtime();

            if(TEST == 1){
                char hostbuffer[100];
                char *IPbuffer;
                struct hostent *host_entry;
                int hostname = gethostname(hostbuffer, sizeof(hostbuffer));

                host_entry = gethostbyname(hostbuffer);

                IPbuffer = inet_ntoa(*((struct in_addr*)
                host_entry->h_addr_list[0]));

                printf("Host name: %s; Host IP: %s; GPU time %lf\n" , hostbuffer, IPbuffer, wend - wstart);

                exit(0);
            }

            REAL E0 = 0.;
            cudaField E0_d;

            E0_d.pitch = pf_d->pitch; E0_d.ptr = pf_d->ptr + 4 * pf_d->pitch*ny*nz;
            ana_residual(E0_d, &E0);

            if(isnan(E0)){
                if(IFLAG_HybridAuto == 1) {
                    //HybridAuto_scheme_IO();
                    //MPI_Barrier(MPI_COMM_WORLD);
                }
                ana_NAN_and_NT();
            }

            REAL T0 = 0.;
            cudaField T0_d;

            T0_d.pitch = pdu_d->pitch; T0_d.ptr = pdu_d->ptr;
            get_inner(T0_d, *pT_d);
            ana_residual(T0_d, &T0);

            if(my_id == 0){
                printf("%lf of %lf ( \033[33m%d\033[0m of %d ) , using \033[36m%lf\033[0m\n", tt , end_time , Istep , end_step  , wend - wstart0);
                printf("%d steps GPU time %lf\n" ,Kstep_show , wend - wstart);
                printf("Averaged Total Energy is %lf\n", E0);
                printf("Averaged Total T is %lf\n", T0);
                printf("\n");
            }
            wstart = MPI_Wtime();
        }

        
        // -----------save data---------------------------------------------
        if(Istep%Kstep_save == 0){
            memcpy_All(pd , pd_d->ptr , pd_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
            memcpy_All(pu , pu_d->ptr , pu_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
            memcpy_All(pv , pv_d->ptr , pv_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
            memcpy_All(pw , pw_d->ptr , pw_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
            memcpy_All(pT , pT_d->ptr , pT_d->pitch , D2H , nx_2lap , ny_2lap , nz_2lap);
            OCFD_save(0, Istep , pd , pu , pv , pw , pT);
        }
        if(end_time <= 0.0) break; //end_time .le. 0  means that stop computation just after saving files

    } while (tt < end_time);
    // --------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_id == 0)
    {
        wend = MPI_Wtime();
        printf("OK! opencfd is finished\n");
        printf("Total GPU time %lf\n" , wend - wstart0);
    }
}


#ifdef __cplusplus
}
#endif

#endif

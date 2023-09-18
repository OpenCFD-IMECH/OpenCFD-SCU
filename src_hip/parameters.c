#ifndef __PARAMETER_H
#define __PARAMETER_H
#include "mpi.h"
#include "pthread.h"
#include "config_parameters.h"
#include "OCFD_Schemes_hybrid_auto.h"
#include "OCFD_Schemes_Choose.h"
#include "utility.h"
#include "stdio.h"
#include "string.h"
#include "assert.h"

#ifdef __cplusplus
extern "C"{
#endif

// ------For Doubleprecision  (real*8)------------------------------------------------------------------
int OCFD_REAL_KIND=8;
MPI_Datatype OCFD_DATA_TYPE = MPI_DOUBLE;   //double precison computing
//  ------For Single precision (real*4)-----------------------
// typedef float REAL;
// int OCFD_REAL_KIND=4,  
// MPI_Datatype OCFD_DATA_TYPE = MPI_REAL;             //  single precision computing
// ===========  Parameters for MPI ==========================================================----------------



// -----constant-----
REAL Re , Pr , Ama , Gamma , Ref_T , epsl_SW , PI = 3.141592653589793;
REAL Cv , Cp , Tsb , amu_C0;
REAL split_C1 , split_C3, tmp0;
REAL vis_flux_init_c;


// --------MPI-------------------
int my_id,npx,npy,npz;  //全局即方向id
int NPX0=0 , NPY0=0 , NPZ0=0; // proc number on each direction
int ID_XP1,ID_XM1,ID_YP1,ID_YM1,ID_ZP1,ID_ZM1; //邻居全局id

MPI_Status status;
MPI_Comm MPI_COMM_X,MPI_COMM_Y,MPI_COMM_Z,MPI_COMM_XY,MPI_COMM_XZ,MPI_COMM_YZ;
MPI_Datatype TYPE_LAPX1,TYPE_LAPY1,TYPE_LAPZ1,TYPE_LAPX2,TYPE_LAPY2,TYPE_LAPZ2;

int *i_offset,*j_offset,*k_offset,*i_nn,*j_nn,*k_nn; //某个方向的分块信息
int MSG_BLOCK_SIZE;

unsigned int nx=0,ny=0,nz=0; // 某方向所处理的个数
unsigned int NX_GLOBAL=0,NY_GLOBAL=0,NZ_GLOBAL=0;
unsigned int nx_lap,ny_lap,nz_lap;
unsigned int nx_2lap,ny_2lap,nz_2lap;

int Stream_MODE; //Stream 模式
int TEST;
pthread_t* thread_handles;
// --------------------------------------------------------------------------------------------------------
REAL dt,end_time,tt;
REAL cpu_time;
int Istep , end_step;

// -----------Analysis and Save------------------------------------------

int OCFD_ANA_time_average; 
int OCFD_ana_flatplate,  OCFD_ana_saveplaneYZ;
int OCFD_ana_saveplaneXZ, OCFD_ana_savePoints;
int OCFD_ana_saveplaneXY, OCFD_ana_saveblock;
int OCFD_ana_getQ;

int Kstep_save, Kstep_show,N_ana,*K_ana,*Kstep_ana,KRK;

//------------Scheme_choose invis and vis----------------------------------------
int Scheme_invis_ID = 0;
int Scheme_vis_ID = 0;

//---------------------WENO_SYMBO_Limiter------------------------------------------
REAL WENO_TV_Limiter = 5.0;
REAL WENO_TV_MAX = 0.2;

// -----------Boundary Condition and Initial Condition-----------------------------
int Iperiodic[3], Jacbound[3], D0_bound[6];
int Non_ref[6];

int Init_stat;
int IBC_USER;


REAL N_nan;
REAL *BC_rpara, (*ANA_rpara)[100]; 
int *BC_npara, (*ANA_npara)[100]; 


// ------Filter----------------------------------------------- 
int FILTER_1,FILTER_2;   // 1: 5-point filter, 2: 7-point wide-band filter
int NF_max = 10;   // maximum of filtering 
int Filter_Fo9p=1, Filter_Fopt_shock=2;
int NFiltering,(*Filter_para)[11];      //Filtering
REAL (*Filter_rpara)[3];   // s0  (0<s0<1), rth
char IF_Filter_X = 0 , IF_Filter_Y  = 0, IF_Filter_Z = 0 ;
int fiter_judge_X = 0, fiter_judge_Y = 0, fiter_judge_Z = 0;
// --------------------------------------------------



// Coordinate parameters 
REAL hx,hy,hz;
REAL *pAxx,*pAyy,*pAzz,*pAkx,*pAky,*pAkz,*pAix,*pAiy,*pAiz,*pAsx,*pAsy,*pAsz,*pAjac;


// calculate memory
REAL *pAmu; // viscous 3d [nz][nt][nx]
REAL *pd,*pu,*pv,*pw,*pT,*pP; //  [nz+2*LAP][ny+2*LAP][nx+2*LAP]
REAL *pf,*pfn,*pdu; // [5][nz][ny][nx]

// used in filtering
REAL *pf_lap; // [nz+2*LAP][ny+2*LAP][nx+2*LAP][5]

// used in invis jacobian , is part of ptmpa
REAL *pfp; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]
REAL *pfm; // [5][nz+2*LAP][ny+2*LAP][nx+2*LAP]
REAL *pcc; // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in invis jacobian , is part of ptmpb
REAL *pdfp , *pdfm; // [nz][ny][nx]

// used in ana
REAL *pQ , *pLamda2;// [nx][ny][nx]
REAL *pdm, *pum, *pvm, *pwm, *pTm;//[nx+2*LAP][ny+2*LAP][nz+2*LAP]
int average_IO = 1;
int Istep_average;
REAL tt_average;

// used in vis jacobian , is part of ptmpb
REAL * pEv1,*pEv2,*pEv3,*pEv4;  // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
// used in vis jacobian , is part of ptmpb
REAL *puk,*pui,*pus,*pvk,*pvi,*pvs,*pwk,*pwi,*pws,*pTk,*pTi,*pTs;  //[nz][ny][nx]
// used in mecpy
REAL *pack_send_x,* pack_recv_x;
REAL *pack_send_y,* pack_recv_y;
REAL *pack_send_z,* pack_recv_z;

// used in boudary_liftbody*********************************************************************

int OCFD_BC_Liftbody3d;
int IF_SYMMETRY;
int IF_WITHLEADING;  // 0 不含头部， 1 含头部 
int IFLAG_UPPERBOUNDARY; // 0 激波外； 1 激波
REAL AOA,TW,EPSL_WALL,EPSL_UPPER,WALL_DIS_BEGIN,WALL_DIS_END;
REAL Sin_AOA , Cos_AOA;


// used in boundary_compressible_conner***********************************************************
int MZMAX, MTMAX, INLET_BOUNDARY, IFLAG_WALL_NOT_NORMAL;
REAL EPSL, X_DIST_BEGIN, X_DIST_END, BETA;
REAL X_WALL_BEGIN, X_UP_BOUNDARY_BEGIN;


// used in SCHEME_HYBRIDAUTO ********************************************************************
int IFLAG_HybridAuto = 0;
int HybridA_Stage = 3, Patch_max = 10;
int IFLAG_mem = 1;
HybridAuto_TYPE HybridAuto;
int *scheme_x, *scheme_y, *scheme_z;

int IF_CHARTERIC;



configItem configList[27] = {
    {"GRID_3D", 0},                 //0
    {"PARALLEL_3D", 0},             //1
    {"LAP", 0},                     //2
    {"MSG_BLOCK_SIZE", 0},          //3      
    {"STREAM", 0},                  //4
    {"TEST", 0},                    //5
    {"IPERIODIC", 0},               //6
    {"JAC_BOUND", 0},               //7
    {"DIF_BOUND", 0},               //8
    {"NON_REFLETION", 0},           //9
    {"SCHEME_INVIS", 0},            //10
    {"SCHEME_VIS", 0},              //11
    {"RE", 0},                      //12
    {"AMA", 0},                     //13
    {"GAMMA", 0},                   //14
    {"PR", 0},                      //15
    {"T_REF", 0},                   //16
    {"EPSL_SW", 0},                 //17
    {"DT", 0},                      //18
    {"END_TIME", 0},                //19
    {"KSTEP_SHOW", 0},              //20
    {"KSTEP_SAVE", 0},              //21
    {"INIT_STAT", 0},               //22
    {"IBC", 0},                     //23
    {"BC_NPARA", 0},                //24
    {"BC_RPARA", 0},                //25
    {"CHARTERIC", 0}                //26   
};


void read_parameters(){
//-------------------------------------------
    int dummy_i, tmp;
    REAL dummy_r;
    int configNum = sizeof(configList)/sizeof(configItem);
    char Scheme_invis[50], Scheme_vis[50];
    char Part_buff[50][1000];

    if(my_id == 0){
        int nk,nr;
        FILE * file = fopen("opencfd-scu.in","r");

        if(file == NULL){
            printf("\033[31mopencfd-scu.in is not find!\033[0m\n");
            exit(-1);
        }

        SearchItem(file, configList, configNum);
        
        sscanf(configList[0].value,"%d%d%d",&NX_GLOBAL,&NY_GLOBAL,&NZ_GLOBAL);
        sscanf(configList[1].value,"%d%d%d",&NPX0,&NPY0,&NPZ0);
        sscanf(configList[2].value,"%d",&dummy_i);
        sscanf(configList[3].value,"%d",&MSG_BLOCK_SIZE);
        sscanf(configList[4].value,"%d",&Stream_MODE);
        sscanf(configList[5].value,"%d",&TEST);

        sscanf(configList[6].value,"%d%d%d",&Iperiodic[0],&Iperiodic[1],&Iperiodic[2]);
        sscanf(configList[7].value,"%d%d%d",&Jacbound[0],&Jacbound[1],&Jacbound[2]);
        sscanf(configList[8].value,"%d%d%d%d%d%d",&D0_bound[0],&D0_bound[1],&D0_bound[2],&D0_bound[3],&D0_bound[4],&D0_bound[5]);
        sscanf(configList[9].value,"%d%d%d%d%d%d",&Non_ref[0],&Non_ref[1],&Non_ref[2],&Non_ref[3],&Non_ref[4],&Non_ref[5]);

        sscanf(configList[10].value,"%s", Scheme_invis);
        sscanf(configList[11].value,"%s", Scheme_vis);
        SCHEME_CHOOSE scheme = {Scheme_invis, Scheme_vis};
        Schemes_Choose_ID(&scheme);

        if(strcmp(Scheme_invis, "SCHEME_HYBRIDAUTO") == 0) IFLAG_HybridAuto = 1;

        HybridAuto.Num_Patch_zones = 0;
        HybridAuto.IF_Smooth_dp = 0;

        HybridAuto.P_intvs = (REAL *)malloc((HybridA_Stage - 1)*sizeof(REAL));
        HybridAuto.zones = (int *)malloc(6*Patch_max*sizeof(int));
        HybridAuto.Pa_zones = (REAL *)malloc(Patch_max*sizeof(REAL));

        if(IFLAG_HybridAuto == 1){

            int (*HybridAuto_zones)[6] = (int(*)[6])HybridAuto.zones;

            configItem Hybridbuff = {"HY_DP_INTV", 0};
            SearchItem(file, &Hybridbuff, 1);

            tmp = PartItem(Hybridbuff.value, Part_buff);
            for(int i=0;i<(HybridA_Stage-1);i++) sscanf(Part_buff[i],"%lf",&HybridAuto.P_intvs[i]);

            sprintf(Hybridbuff.name, "HY_STYLE");
            SearchItem(file, &Hybridbuff, 1);
            sscanf(Hybridbuff.value,"%d",&HybridAuto.Style);

            if(HybridAuto.Style != 1 && HybridAuto.Style != 2){
                printf("\033[31mHYBRID SCHEMES CHOOSE IS WRONG！！！\033[0m\n");
                exit(0);
            }

            sprintf(Hybridbuff.name, "HY_SMOOTH_DP");
            SearchItem(file, &Hybridbuff, 1);
            sscanf(Hybridbuff.value,"%d",&HybridAuto.IF_Smooth_dp);

            sprintf(Hybridbuff.name, "HY_PATCH_ZONE");
            SearchItem(file, &Hybridbuff, 1);
            sscanf(Hybridbuff.value,"%d",&HybridAuto.Num_Patch_zones);

            for(int i=0; i<HybridAuto.Num_Patch_zones; i++){
                sprintf(Hybridbuff.name, "HY_ZONE%d", i);
                SearchItem(file, &Hybridbuff, 1);

                sscanf(Hybridbuff.value,"%d%d%d%d%d%d%lf",&HybridAuto_zones[i][0],&HybridAuto_zones[i][1],&HybridAuto_zones[i][2],
                &HybridAuto_zones[i][3],&HybridAuto_zones[i][4],&HybridAuto_zones[i][5],&HybridAuto.Pa_zones[i]);
            }
        }

        sscanf(configList[12].value,"%lf",&Re);
        sscanf(configList[13].value,"%lf",&Ama);
        sscanf(configList[14].value,"%lf",&Gamma);
        sscanf(configList[15].value,"%lf",&Pr);
        sscanf(configList[16].value,"%lf",&Ref_T);
        sscanf(configList[17].value,"%lf",&epsl_SW);

        sscanf(configList[18].value,"%lf",&dt);
        sscanf(configList[19].value,"%lf",&end_time);
        sscanf(configList[20].value,"%d",&Kstep_show);
        sscanf(configList[21].value,"%d",&Kstep_save);
        sscanf(configList[22].value,"%d",&Init_stat);

        sscanf(configList[23].value,"%d",&IBC_USER);
        BC_npara = (int*)malloc(sizeof(int)*100);
        BC_rpara = (REAL*)malloc(sizeof(REAL)*100);

        nk = PartItem(configList[24].value, Part_buff);
        for(int i=0;i<nk;i++) sscanf(Part_buff[i],"%d",BC_npara+i);
        
        nr = PartItem(configList[25].value, Part_buff);
        for(int i=0;i<nr;i++) sscanf(Part_buff[i],"%lf",BC_rpara+i);

        sscanf(configList[26].value,"%d", &IF_CHARTERIC);

        int NameNUM[1000];

        NFiltering = ItemNUM(file, "FILTER_NPARA", &NameNUM[0]);

        Filter_para = (int(*)[11])malloc(sizeof(int)*(NFiltering+1)*11);
        Filter_rpara = (REAL(*)[3])malloc(sizeof(REAL)*(NFiltering+1)*3);

        for(int i=0;i<NFiltering;i++){
            configItem Hybridbuff;
            //ntime, Filter_X, Filter_Y, Filter_Z, ib, ie, jb, je, kb, ke, Filter_scheme
            sprintf(Hybridbuff.name, "FILTER_NPARA%d", NameNUM[i]);
            SearchItem(file, &Hybridbuff, 1);

            tmp = PartItem(Hybridbuff.value, Part_buff);
            for(int n=0;n<11;n++) sscanf(Part_buff[n],"%d",&Filter_para[i][n]);
            for(int n=0;n<11;n++) Filter_para[i+1][n] = Filter_para[i][n];

            sprintf(Hybridbuff.name, "FILTER_RPARA%d", NameNUM[i]);
            SearchItem(file, &Hybridbuff, 1);

            tmp = PartItem(Hybridbuff.value, Part_buff);
            for(int n=0;n<3;n++) sscanf(Part_buff[n],"%lf",&Filter_rpara[i][n]);
            for(int n=0;n<3;n++) Filter_rpara[i+1][n] = Filter_rpara[i][n];
        }

        N_ana = ItemNUM(file, "ANA_EVENT", &NameNUM[0]);

        ANA_npara = (int(*)[100])malloc(sizeof(int)*100*N_ana);
        ANA_rpara = (REAL(*)[100])malloc(sizeof(REAL)*100*N_ana);
        K_ana = (int*)malloc(sizeof(int)*N_ana);
        Kstep_ana = (int*)malloc(sizeof(int)*N_ana);

        for(int i=0;i<N_ana;i++){
            configItem Hybridbuff;
            sprintf(Hybridbuff.name, "ANA_EVENT%d", NameNUM[i]);
            SearchItem(file, &Hybridbuff, 1);

            sscanf(Hybridbuff.value,"%d%d",K_ana+i,Kstep_ana+i);

            sprintf(Hybridbuff.name, "ANA_NPARA%d", NameNUM[i]);
            SearchItem(file, &Hybridbuff, 1);

            nk = PartItem(Hybridbuff.value, Part_buff);
            for(int n=0;n<nk;n++) sscanf(Part_buff[n],"%d",&ANA_npara[i][n]);

            sprintf(Hybridbuff.name, "ANA_RPARA%d", NameNUM[i]);
            SearchItem(file, &Hybridbuff, 1);

            nr = PartItem(Hybridbuff.value, Part_buff);
            for(int n=0;n<nr;n++) sscanf(Part_buff[n],"%lf",&ANA_rpara[i][n]);
        }

        fclose(file);

    } else {
        int nk = 100;
        int nr = 100;
        
        N_ana = 10;
        
        BC_npara = (int*)malloc(sizeof(int)*nk);
        BC_rpara = (REAL*)malloc(sizeof(REAL)*nr);

        Filter_para = (int(*)[11])malloc(sizeof(int)*NF_max*11);
        Filter_rpara = (REAL(*)[3])malloc(sizeof(REAL)*NF_max*3);

        ANA_npara = (int(*)[100])malloc(sizeof(int)*100*N_ana);
        ANA_rpara = (REAL(*)[100])malloc(sizeof(REAL)*100*N_ana);
        K_ana = (int*)malloc(sizeof(int)*N_ana);
        Kstep_ana = (int*)malloc(sizeof(int)*N_ana);

        HybridAuto.P_intvs = (REAL *)malloc((HybridA_Stage - 1)*sizeof(REAL));
        HybridAuto.zones = (int *)malloc(6*Patch_max*sizeof(int));
        HybridAuto.Pa_zones = (REAL *)malloc(Patch_max*sizeof(REAL));
    }

    int btmp[18];
    if(my_id == 0){
        btmp[0]=Jacbound[0];
        btmp[1]=Jacbound[1];
        btmp[2]=Jacbound[2];

        btmp[3]=D0_bound[0];
        btmp[4]=D0_bound[1];
        btmp[5]=D0_bound[2];
        btmp[6]=D0_bound[3];
        btmp[7]=D0_bound[4];
        btmp[8]=D0_bound[5];

        btmp[9] =Iperiodic[0];
        btmp[10]=Iperiodic[1];
        btmp[11]=Iperiodic[2];

        for(int i = 0; i < 6; i++){
            btmp[i+12] = Non_ref[i];
        }
    }

       MPI_Bcast(btmp, 18, MPI_INT, 0, MPI_COMM_WORLD);

    if(my_id!=0){
        Jacbound[0]=btmp[0];
        Jacbound[1]=btmp[1];
        Jacbound[2]=btmp[2];

        D0_bound[0]=btmp[3];
        D0_bound[1]=btmp[4];
        D0_bound[2]=btmp[5];
        D0_bound[3]=btmp[6];
        D0_bound[4]=btmp[7];
        D0_bound[5]=btmp[8];

        Iperiodic[0]=btmp[9];
        Iperiodic[1]=btmp[10];
        Iperiodic[2]=btmp[11];

        for(int i = 0; i < 6; i++){
            Non_ref[i] = btmp[i+12];
        }
    }
   
//     Boardcast integer and real parameters to all proc    
    int ntmp[19];
    if(my_id == 0){
        ntmp[0]=NX_GLOBAL;
        ntmp[1]=NY_GLOBAL;
        ntmp[2]=NZ_GLOBAL;
        ntmp[3]=NPX0;
        ntmp[4]=NPY0;
        ntmp[5]=NPZ0;
        ntmp[6]=LAP;
        ntmp[7]=MSG_BLOCK_SIZE;               // defined in opencfd.h, as in common /ocfd_mpi_comm/  
        ntmp[8]=Kstep_show;
        ntmp[9]=IBC_USER;
        ntmp[10]=N_ana;
        ntmp[11]=Kstep_save;
        ntmp[12]=Init_stat;
        ntmp[13]=NFiltering;
        ntmp[14]=Scheme_invis_ID;
        ntmp[15]=Scheme_vis_ID;
        ntmp[16]=Stream_MODE;
        ntmp[17]=TEST;
        ntmp[18]=IF_CHARTERIC;
    }

	MPI_Bcast(ntmp, 19, MPI_INT, 0, MPI_COMM_WORLD);

    if(my_id!=0){
        NX_GLOBAL = ntmp[0];
        NY_GLOBAL = ntmp[1];
        NZ_GLOBAL = ntmp[2];
        NPX0 = ntmp[3];
        NPY0 = ntmp[4];
        NPZ0 = ntmp[5];
        dummy_i = ntmp[6];
        MSG_BLOCK_SIZE = ntmp[7];                 // defined in opencfd.h, as in common /ocfd_mpi_comm/  
        Kstep_show = ntmp[8];
        IBC_USER = ntmp[9];
        N_ana = ntmp[10];
        Kstep_save = ntmp[11];
        Init_stat = ntmp[12];
        NFiltering = ntmp[13];
        Scheme_invis_ID = ntmp[14];
        Scheme_vis_ID = ntmp[15];
        Stream_MODE = ntmp[16];
        TEST = ntmp[17];
        IF_CHARTERIC = ntmp[18];
    }

//c----------------------------------------------------
    REAL rtmp[8];
    if(my_id==0){
        rtmp[0]=Re;
        rtmp[1]=Ama;
        rtmp[2]=Gamma;
        rtmp[3]=Pr;
        rtmp[4]=dt;
        rtmp[5]=end_time;
        rtmp[6]=Ref_T;
        rtmp[7]=epsl_SW;            // epsl in Steger-Warming splitting
    }

    MPI_Bcast(rtmp , 8 , OCFD_DATA_TYPE , 0 , MPI_COMM_WORLD);
    
    if(my_id!=0){
        Re = rtmp[0];
        Ama = rtmp[1];
        Gamma = rtmp[2];
        Pr = rtmp[3];
        dt = rtmp[4];
        end_time = rtmp[5];
        Ref_T = rtmp[6];
        epsl_SW = rtmp[7];
    }
    if(Ref_T <= 0) Ref_T=288.150;

	dummy_i=5;

    int htmp[3];

    if(my_id == 0){
        htmp[0] = IFLAG_HybridAuto;
        htmp[1] = HybridAuto.Num_Patch_zones;
        htmp[2] = HybridAuto.IF_Smooth_dp;
        htmp[3] = HybridAuto.Style;
    }

    MPI_Bcast(htmp, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if(my_id != 0){
        IFLAG_HybridAuto = htmp[0];
        HybridAuto.Num_Patch_zones = htmp[1];
        HybridAuto.IF_Smooth_dp = htmp[2];
        HybridAuto.Style = htmp[3];
    }

    MPI_Bcast(HybridAuto.P_intvs, HybridA_Stage - 1, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
    MPI_Bcast(HybridAuto.zones, 6*Patch_max, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(HybridAuto.Pa_zones, Patch_max, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
//c----------------------------------------------------------------------------

    MPI_Bcast(K_ana, N_ana, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Kstep_ana, N_ana, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(BC_npara, 100, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(BC_rpara, 100, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);

    MPI_Bcast(ANA_npara, N_ana*100, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ANA_rpara, N_ana*100, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);

    MPI_Bcast(Filter_para, 11*(NFiltering+1), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Filter_rpara, 3*(NFiltering+1), OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);

//---------------------------------------------------------------------------------
//   print the parameters
  if(my_id == 0){
    printf("##################################################################################################\n");
    printf("Welcome to use OpenCFD-SCU-V1.00!\nCopyRight by Li-Xinliang, LHD, Institute of Mechanics, CAS (lixl@imech.ac.cn)\n");
    printf("Coded by Liu-Shiwei, ICMSEC, Academy of Mathematics and Systems Science, CAS (liusw@lsec.cc.ac.cn)\n");
    printf("Coded by Dang-Guanlin, LHD, Institute of Mechanics, CAS (dangguanlin@imech.ac.cn) 2020-01\n");
    printf("Mesh(Nx,Ny,Nz): (%d,%d,%d)\n" , NX_GLOBAL, NY_GLOBAL, NZ_GLOBAL);
    printf("3D Partation: %d*%d*%d   Total procs=%d\n", NPX0,NPY0,NPZ0 , NPX0*NPY0*NPZ0);
    printf("Re=%f , Ma=%f , Gamma=%f , dt=%f\n", Re, Ama, Gamma, dt);
    if(IFLAG_HybridAuto == 1) printf("Hybrid Scheme enabled, Hybrid style is %d\n", HybridAuto.Style);
    printf("Start Computing ......\n");
 
    FILE * file;
    file = fopen("opencfd.log","a"); 
    fprintf(file,"##################################################################################################\n");
    fprintf(file,"OpenCFD-SCU-V1.00 CopyRight by Li-Xinliang, LHD, Institute of Mechanics, CAS (lixl@imech.ac.cn)\n");
    fprintf(file,"Coded by Liu-Shiwei, ICMSEC, Academy of Mathematics and Systems Science, CAS (liusw@lsec.cc.ac.cn)\n");
    fprintf(file,"Coded by Dang-Guanlin, LHD, Institute of Mechanics, CAS (dangguanlin@imech.ac.cn) 2020-01\n");
    fprintf(file,"Mesh(Nx,Ny,Nz): (%d,%d,%d)\n" , NX_GLOBAL, NY_GLOBAL, NZ_GLOBAL);
    fprintf(file,"3D Partation: %d*%d*%d   Total procs=%d\n", NPX0,NPY0,NPZ0 , NPX0*NPY0*NPZ0);
    fprintf(file,"Re=%f , Ma=%f , Gamma=%f , dt=%f\n", Re, Ama, Gamma, dt);
    if(IFLAG_HybridAuto == 1) fprintf(file, "Hybrid Scheme enabled, Hybrid style is %d\n", HybridAuto.Style);
    fprintf(file,"Start Computing ......\n");

    fclose(file);
  }
//-------------------------------------------------------
}


void* vis_choose_CD6(void* Scheme_vis){
    char *Scheme_VIS = (char *)Scheme_vis;
    if(strcmp(Scheme_VIS, "CD6") == 0) Scheme_vis_ID = 203;
}

void* vis_choose_CD8(void* Scheme_vis){
    char *Scheme_VIS = (char *)Scheme_vis;
    if(strcmp(Scheme_VIS, "CD8") == 0) Scheme_vis_ID = 204;
}
//----------------------------------------------------------------------
void* invis_choose_up7(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "UP7") == 0) Scheme_invis_ID = 301;
}

void* invis_choose_weno5(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "WENO5") == 0) Scheme_invis_ID = 302;
}

void* invis_choose_weno7(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "WENO7") == 0) Scheme_invis_ID = 303;
}

void* invis_choose_weno7_symbo(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "WENO7_SYMBO") == 0) Scheme_invis_ID = 304;
}

void* invis_choose_weno7_symbo_limit(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "WENO7_SYMBO_LIM") == 0) Scheme_invis_ID = 305;
}

void* invis_choose_NND2(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "NND2") == 0) Scheme_invis_ID = 306;
}

void* invis_choose_OMP6_HR(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "OMP6_HR") == 0) Scheme_invis_ID = 307;  //for OMP6, High-robust
}

void* invis_choose_OMP6_LD(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "OMP6_LD") == 0) Scheme_invis_ID = 308;  //for OMP6, Low-dissipation
}

void* invis_choose_OMP6_CD8(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "OMP6_CD8") == 0) Scheme_invis_ID = 309;  //for OMP6, 8th-Center
}

void* invis_choose_SCHEME_HYBRIDAUTO(void* Scheme_invis){
    char *Scheme_INVIS = (char *)Scheme_invis;
    if(strcmp(Scheme_INVIS, "SCHEME_HYBRIDAUTO") == 0) Scheme_invis_ID = 310;
}


void Schemes_Choose_ID(SCHEME_CHOOSE *scheme){

    pthread_create(&thread_handles[0], NULL, invis_choose_weno7_symbo, (void*)(*scheme).invis);
    pthread_create(&thread_handles[1], NULL, invis_choose_weno7_symbo_limit, (void*)(*scheme).invis);
    pthread_create(&thread_handles[2], NULL, invis_choose_weno5, (void*)(*scheme).invis);
    pthread_create(&thread_handles[3], NULL, invis_choose_weno7, (void*)(*scheme).invis);
    pthread_create(&thread_handles[4], NULL, invis_choose_NND2, (void*)(*scheme).invis);
    pthread_create(&thread_handles[5], NULL, invis_choose_OMP6_HR, (void*)(*scheme).invis);
    pthread_create(&thread_handles[6], NULL, invis_choose_OMP6_LD, (void*)(*scheme).invis);
    pthread_create(&thread_handles[7], NULL, invis_choose_OMP6_CD8, (void*)(*scheme).invis);
    pthread_create(&thread_handles[8], NULL, invis_choose_up7, (void*)(*scheme).invis);

    pthread_create(&thread_handles[9], NULL, invis_choose_SCHEME_HYBRIDAUTO, (void*)(*scheme).invis);


    pthread_create(&thread_handles[10], NULL, vis_choose_CD6, (void*)(*scheme).vis);
    pthread_create(&thread_handles[11], NULL, vis_choose_CD8, (void*)(*scheme).vis);

    for(long thread = 0; thread < 12; thread++)
        pthread_join(thread_handles[thread], NULL);

    if(Scheme_invis_ID == 0 || Scheme_vis_ID == 0){
        printf("\033[31mSCHEMES CHOOSE IS WRONG！！！\033[0m\n");
        exit(0);
    }
}

int ExtarctItem(char *src, char *name, char *value){
    char *eq, *lf;
    eq = strchr(src, '=');
    lf = strchr(src, '\n');

    if(eq != NULL && lf != NULL){
        *lf = '\0';
        strncpy(name, src, eq-src);
        strcpy(value, eq+1);
        return 1;
    }

    return 0;
}


void ModifyItem(char *name, char *buff){
    while(*name != '\0')
    {
        if(*name != ' '){
            *buff = *name;
            buff++;
        }
        name++;
    }
}

void RemovalNUM(char *buff){
    int i, j;

    for(i=j=0; buff[i]!='\0'; i++){
        if(buff[i]<'0' || buff[i]>'9')
            buff[j++] = buff[i];
    }

    buff[j] = '\0';
}

int StringToInteger(char *buff){
    int value = 0;

    while(*buff != '\0')
    {
        if(*buff>='0' && *buff<='9')
            value = value*10 + *buff - '0';

        buff++;
    }
    
    return value;
}


void SearchItem(FILE *file, configItem *List, int configNum){
    int N = 1000;
    char buff[N];
    char name[N];
    char value[N];

    rewind(file);

    while(fgets(buff, N, file))
    {
        if(ExtarctItem(buff, name, value)){
            memset(buff, 0, strlen(buff));
            ModifyItem(name, buff);
            for(int i = 0; i < configNum; i++){
                if(strcmp(buff, List[i].name) == 0){
                    strcpy(List[i].value, value);
                }
            }
            memset(name, 0, strlen(name));
        }
    }
}


int ItemNUM(FILE *file, char *Item_name, int *NameNUM){
    int N = 1000;
    int i = 0;
    char buff[N];
    char name[N];
    char value[N];

    rewind(file);

    while(fgets(buff, N, file))
    {
        if(ExtarctItem(buff, name, value)){
            memset(buff, 0, strlen(buff));
            ModifyItem(name, buff);

            RemovalNUM(buff);

            if(strcmp(buff, Item_name) == 0){
                i += 1;
                *NameNUM = StringToInteger(name);
                NameNUM++;
            }

            memset(name, 0, strlen(name));
        }
    }

    return i;
}


int PartItem(char *src, char part[][1000]){
    const char blank[2] = " ";
    int num = 0;

    char *buff;

    buff = strtok(src, blank);

    while (buff != NULL)
    {
        strcpy(part[num], buff);
        buff = strtok(NULL, blank);

        num += 1;
    }

    return num;
}

#ifdef __cplusplus
}
#endif

#endif




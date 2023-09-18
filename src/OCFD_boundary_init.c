#include "math.h"
#include "parameters.h"
#include "utility.h"
#include "OCFD_IO.h"
#include "io_warp.h"
#include "OCFD_init.h"
#include "OCFD_boundary_Liftbody3D.h"
#include "OCFD_Comput_Jacobian3d.h"
#include "OCFD_boundary_init.h"
#include "OCFD_init.h"

#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "commen_kernel.h"
#include "time.h"
#include "mpi.h"

#ifdef __cplusplus
extern "C"{
#endif

// used in boudary_liftbody*********************************************************************
    char v_dist_need = 0;
    char TW_postive = 0;
    REAL *pu2d_inlet; //[5][nz][ny]
    REAL *pu2d_upper; //[5][ny][nx]
        
    REAL * pv_dist_wall; // [ny][nx]
    REAL *pv_dist_coeff; // [3][ny][nx]
    REAL * pu_dist_upper; // [ny][nx]

    cudaField *pu2d_inlet_d; //[5][nz][ny]
    cudaField *pu2d_upper_d; //[5][ny][nx]
    //cudaField *pv_dist_wall_d; // [ny][nx]
    cudaField *pv_dist_coeff_d; // [3][ny][nx]
    cudaField *pu_dist_upper_d; // [ny][nx]
    

// used in boundary_compressible_conner***********************************************************
    int x_begin;
    REAL *pub1; // [ny][4]
    REAL *pfx; // [nx]
    REAL *pgz; // [nz]
    REAL *TM; // [MTMAX]
    REAL *fait; // [MTMAX]
    REAL SLZ;


void bc_parameter(){
    switch(IBC_USER){
        case 124:
        {
            IF_SYMMETRY = BC_npara[0];
	        IF_WITHLEADING = BC_npara[1];		// 0 不含头部， 1 含头部
	        IFLAG_UPPERBOUNDARY = BC_npara[2]; // 0 激波外； 1 激波
            MZMAX = BC_npara[3];
            MTMAX = BC_npara[4];

	        AOA = BC_rpara[0]; // attack angle
            Sin_AOA = sin(AOA * PI/180);
            Cos_AOA = cos(AOA * PI/180);

	        TW = BC_rpara[1]; // Wall temperature
            EPSL_WALL = BC_rpara[2];
            EPSL_UPPER = BC_rpara[3];
            BETA = BC_rpara[4];
	        WALL_DIS_BEGIN = BC_rpara[5];
            WALL_DIS_END = BC_rpara[6];
        }
        break;

        case 108:
        {
            MZMAX = BC_npara[0];
            MTMAX = BC_npara[1];
            INLET_BOUNDARY = BC_npara[2];
            IFLAG_WALL_NOT_NORMAL = BC_npara[3];

            TW = BC_rpara[0];
            EPSL = BC_rpara[1];
            X_DIST_BEGIN = BC_rpara[2];
            X_DIST_END = BC_rpara[3];
            BETA = BC_rpara[4];
            X_WALL_BEGIN = BC_rpara[5];
            X_UP_BOUNDARY_BEGIN = BC_rpara[6];
            SLZ = BC_rpara[7];
        }
        break;

    }
}


void bc_user_Liftbody3d_init()
{

    opencfd_mem_init_boundary();

    // cudaMemcpyToSymbol( HIP_SYMBOL(Sin_AOA_d) , &Sin_AOA , sizeof(REAL) , 0 , cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol( HIP_SYMBOL(Cos_AOA_d) , &Cos_AOA , sizeof(REAL) , 0 , cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol( HIP_SYMBOL(TW_d) , &TW , sizeof(REAL) , 0 , cudaMemcpyHostToDevice);
    
	v_dist_need = 0;
	if(TW > 0.0) TW_postive = 1; else TW_postive = 0;

	int i, j, k, m;

	REAL(*u2d_inlet)[nz][ny] = (REAL(*)[nz][ny])pu2d_inlet; //[5][nz][ny]
	REAL(*u2d_upper)[ny][nx] = (REAL(*)[ny][nx])pu2d_upper; //[5][ny][nx]

	REAL(*v_dist_wall)[nx]  = (REAL(*)[nx])pv_dist_wall;
	REAL(*u_dist_upper)[nx] = (REAL(*)[nx])pu_dist_upper;

	//--------------------------------------------------------------------------

	// amplitude of wall blow and suction disturbance


	REAL tmp2d[NZ_GLOBAL][NY_GLOBAL];  //NY_GLOBAL,NZ_GLOBAL

	//--------------Inlet boundary condition ---------------------
	if (IF_WITHLEADING == 1)
	{
		FILE *file;
		if (my_id == 0)
		{
			file = fopen("flow-inlet-section.dat", "r");
			printf("read inlet boundary data: flow-inlet-section.dat\n");
		}
		int j1, k1;
		for (m = 0; m < 5; m++)
		{
			if (my_id == 0) FREAD(tmp2d, sizeof(REAL), NZ_GLOBAL * NY_GLOBAL, file)
			MPI_Bcast(tmp2d, NY_GLOBAL * NZ_GLOBAL, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
			for (k = 0; k < nz; k++)
			{
				for (j = 0; j < ny; j++)
				{
					k1 = k_offset[npz] + k;
					j1 = j_offset[npy] + j;
					u2d_inlet[m][k][j] = tmp2d[k1][j1];
				}
			}
		}

		if (my_id == 0)
			fclose(file);

	}else{
        REAL (*d)[ny+2*LAP][nx+2*LAP] = (REAL(*)[ny+2*LAP][nx+2*LAP])pd;
        REAL (*u)[ny+2*LAP][nx+2*LAP] = (REAL(*)[ny+2*LAP][nx+2*LAP])pu;
        REAL (*v)[ny+2*LAP][nx+2*LAP] = (REAL(*)[ny+2*LAP][nx+2*LAP])pv;
        REAL (*w)[ny+2*LAP][nx+2*LAP] = (REAL(*)[ny+2*LAP][nx+2*LAP])pw;
        REAL (*T)[ny+2*LAP][nx+2*LAP] = (REAL(*)[ny+2*LAP][nx+2*LAP])pT;

        for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				u2d_inlet[0][k][j] = d[k+LAP][j+LAP][LAP];
				u2d_inlet[1][k][j] = u[k+LAP][j+LAP][LAP];
				u2d_inlet[2][k][j] = v[k+LAP][j+LAP][LAP];
				u2d_inlet[3][k][j] = w[k+LAP][j+LAP][LAP];
				u2d_inlet[4][k][j] = T[k+LAP][j+LAP][LAP];
			}
		}
    }

	memcpy_All(pu2d_inlet, pu2d_inlet_d->ptr, pu2d_inlet_d->pitch, H2D, ny, nz, 5);

    //if (Init_stat == 2) liftbody_init();
	//----------Upper boundary conditon -----------------
	if (IFLAG_UPPERBOUNDARY == 1)
	{
		REAL tmp2d1[NY_GLOBAL][NX_GLOBAL]; //NX_GLOBAL,NY_GLOBAL
		FILE *file;
		if (my_id == 0)
		{
			file = fopen("flow-outboundary.dat", "r");
			printf("read upper boundary data: flow-outboundary.dat\n");
		}
		int j1, i1;
		for (m = 0; m < 5; m++)
		{
			if (my_id == 0) FREAD(tmp2d1, sizeof(REAL), NX_GLOBAL * NY_GLOBAL, file)
			MPI_Bcast(tmp2d1, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					j1 = j_offset[npy] + j;
					i1 = i_offset[npx] + i;
					u2d_upper[m][j][i] = tmp2d1[j1][i1];
				}
			}
		}
		memcpy_All(pu2d_upper , pu2d_upper_d->ptr , pu2d_upper_d->pitch , H2D , nx,ny,5);

		if (my_id == 0)
			fclose(file);
	}

	//------random wall disturbance ------------------------------
	REAL(*Axx)
	[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pAxx, nx + 2 * LAP, ny + 2 * LAP);
	REAL rand_x;
	srand((unsigned)time(NULL));

    REAL tmp_v_dist[NY_GLOBAL][NX_GLOBAL];
    REAL tmp_u_dist[NY_GLOBAL][NX_GLOBAL];

    REAL(*Ayy)[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pAyy, nx + 2 * LAP, ny + 2 * LAP);
	REAL(*Azz)[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pAzz, nx + 2 * LAP, ny + 2 * LAP);

	REAL(*v_dist_coeff)[ny][nx] = (REAL(*)[ny][nx])pv_dist_coeff;

    int ilap , jlap;

    if(MZMAX != 0){
        v_dist_need = 1;
        if(my_id == 0) printf("Disturbance has been added, MZMAX is %d\n", MZMAX);

        REAL *xa = (REAL*)malloc(sizeof(REAL)*nx);

        for(int i = 0; i < nx; i++){
            xa[i] = Axx[LAP][LAP][i + LAP];
        }

        get_xy_blow_suction_multiwave(nx, MZMAX, xa, pfx, pgz, WALL_DIS_BEGIN, WALL_DIS_END);

        for (j = 0; j < ny; j++)
	    {
	    	for (i = 0; i < nx; i++)
	    	{

		    	v_dist_wall[j][i] = pfx[i]*pgz[j];
    		}
    	}

	    for (j = LAP; j < ny + LAP; j++)
	    {   
            jlap = j-LAP;
	    	for (i = LAP; i < nx + LAP; i++)
	    	{
                ilap =i-LAP;

	    		v_dist_coeff[0][jlap][ilap]= 0;
	    		v_dist_coeff[1][jlap][ilap]= EPSL_WALL * v_dist_wall[jlap][ilap] * sin(2*PI/NY_GLOBAL*(jlap+j_offset[npy]));
	    		v_dist_coeff[2][jlap][ilap]= EPSL_WALL * v_dist_wall[jlap][ilap] * cos(2*PI/NY_GLOBAL*(jlap+j_offset[npy]));
	    	}
	    }

    }else{

	    for (j = 0; j < NY_GLOBAL; j++)
	    {
	    	for (i = 0; i < NX_GLOBAL; i++)
	    	{
	    		rand_x = (rand() / (REAL)RAND_MAX - 0.5) * 2.0;
	    		tmp_v_dist[j][i] = EPSL_WALL * rand_x;

	    		rand_x = (rand() / (REAL)RAND_MAX - 0.5) * 2.0;
	    		tmp_u_dist[j][i] = EPSL_UPPER * rand_x;
	    	}
	    }

        MPI_Bcast(tmp_v_dist, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
        MPI_Bcast(tmp_u_dist, NX_GLOBAL * NY_GLOBAL, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);

	    int j1, i1;
	    for (j = 0; j < ny; j++)
	    {
	    	for (i = 0; i < nx; i++)
	    	{
	    		i1 = i_offset[npx] + i;
                j1 = j_offset[npy] + j;

		    	if (Axx[LAP][j + LAP][i + LAP] >= WALL_DIS_BEGIN && Axx[LAP][j + LAP][i + LAP] <= WALL_DIS_END)
		    	{
		    		v_dist_need = 1;
		    		v_dist_wall[j][i] = tmp_v_dist[j1][i1];
		    	}
		    	else
		    	{
		    		v_dist_wall[j][i] = 0.0;
		    	}

	    		u_dist_upper[j][i] = tmp_u_dist[j1][i1];
	    	}
	    }

	    //memcpy_All(pv_dist_wall  , pv_dist_wall_d->ptr  , pv_dist_wall_d->pitch  , H2D , nx,ny,1);
	    memcpy_All(pu_dist_upper, pu_dist_upper_d->ptr, pu_dist_upper_d->pitch, H2D, nx, ny, 1);

        REAL xn, yn, zn, sn;

	    for (j = LAP; j < ny + LAP; j++)
	    {   jlap = j-LAP;
	    	for (i = LAP; i < nx + LAP; i++)
	    	{
                ilap =i-LAP;
	    		xn = Axx[LAP+1][j][i] - Axx[LAP][j][i];
	    		yn = Ayy[LAP+1][j][i] - Ayy[LAP][j][i];
	    		zn = Azz[LAP+1][j][i] - Azz[LAP][j][i];
	    		sn = sqrt(xn * xn + yn * yn + zn * zn);

	    		v_dist_coeff[0][jlap][ilap]= v_dist_wall[jlap][ilap] * xn / sn;
	    		v_dist_coeff[1][jlap][ilap]= v_dist_wall[jlap][ilap] * yn / sn;
	    		v_dist_coeff[2][jlap][ilap]= v_dist_wall[jlap][ilap] * zn / sn;
	    	}
	    }
    }

    get_fait_multifrequancy(MTMAX);//Comput TM
    
	memcpy_All(pv_dist_coeff, pv_dist_coeff_d->ptr, pv_dist_coeff_d->pitch, H2D, nx, ny, 3);
}

void get_xy_blow_suction_multiwave(int NX, int MZ_MAX, REAL *xx,
REAL *fx, REAL *gz, REAL DIST_BEGIN, REAL DIST_END){
    int MZ_MAX1;
    REAL ztmp, seta;
    REAL *faiz, *zl;
    
    MZ_MAX1 = abs(MZ_MAX);
    faiz = (REAL*)malloc(sizeof(REAL)*MZ_MAX1);
    zl = (REAL*)malloc(sizeof(REAL)*MZ_MAX1);
    
    ztmp = 0.;
    
    for(int k = 0; k < MZ_MAX1; k++){
        faiz[k] = rand()/(REAL)RAND_MAX;
        if(k == 0){
            zl[k] = 1.;
        }else{
            zl[k] = zl[k - 1] / 1.25;
        }
        ztmp = ztmp + zl[k];
    }
    
    for(int k = 0; k < MZ_MAX1; k++){
        zl[k] = zl[k] / ztmp;
    }
    
    MPI_Bcast(faiz, MZ_MAX1, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
    
    for(int i = 0; i < NX; i++){
        if(xx[i] >= DIST_BEGIN && xx[i] <= DIST_END){
            //seta = 2. * PI * (xx[i] - DIST_BEGIN)/(DIST_BEGIN - DIST_END);
            seta = 10. * PI * (xx[i] - DIST_BEGIN)/(DIST_BEGIN - DIST_END);
            fx[i] = 4. / sqrt(27.) * sin(seta) * (1. - cos(seta));
        }else{
            fx[i] = 0.;
        }
    }

    for(int j = 0; j < ny; j++){
        gz[j] = 0.;
        seta = ((REAL)(j + j_offset[npy]))/NY_GLOBAL;
        if(MZ_MAX > 0){
            for(int m = 0; m < MZ_MAX; m++){
                gz[j] = gz[j] + zl[m] * sin(2. * PI * (m + 1) * (seta + faiz[m]));
            }
        }else if(MZ_MAX == 0){
            gz[j] = 1.;
        }else{
            gz[j] = sin(-2. * PI * MZ_MAX * seta);
        }
    }

    free(faiz);
    free(zl);
}

void bc_user_Compression_conner_init(){

    opencfd_mem_init_boundary();

    REAL(*ub1)[ny]= (REAL(*)[ny])pub1;

    FILE *file;
    REAL tmp[NY_GLOBAL][4];

    if(my_id == 0){
        if(INLET_BOUNDARY == 1){
            char str[100];
            REAL tmp1;
            file = fopen("flow1d-inlet.dat", "r");
            printf("read inlet boundary data: flow1d-inlet.dat\n");
            fgets(str, 100, file);
            for(int j = 0; j < NY_GLOBAL; j++){
                fscanf(file, "%lf%lf%lf%lf%lf\n", &tmp1, &tmp[j][0], &tmp[j][1], &tmp[j][2], &tmp[j][3]);
            }
            fclose(file);
        }else{
            for(int j = 0; j < NY_GLOBAL; j++){
                tmp[j][0] = 1.;
                tmp[j][1] = 1.;
                tmp[j][2] = 0.;
                tmp[j][3] = 1.; 
            }
        }
    }

    MPI_Bcast(tmp, 4 * NY_GLOBAL, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);

    int j1;
    REAL *xa, *za;
    xa = (REAL*)malloc(sizeof(REAL)*nx);
    za = (REAL*)malloc(sizeof(REAL)*nz);

    REAL(*Axx)[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pAxx, nx + 2 * LAP, ny + 2 * LAP);
	REAL(*Azz)[ny + 2 * LAP][nx + 2 * LAP] = PTR2ARRAY2(pAzz, nx + 2 * LAP, ny + 2 * LAP);

    for(int j = 0; j < ny; j++){
        j1 = j_offset[npy] + j;
        for(int i = 0; i < 4; i++){
            ub1[i][j] = tmp[j1][i];
        }
    }

    for(int i = 0; i < nx; i++){
        if(Axx[LAP][ny + LAP - 1][i + LAP] <= X_UP_BOUNDARY_BEGIN) x_begin = i; 
        for(int k = 0; k < nz; k++){
            xa[i] = Axx[LAP][LAP][i + LAP];
            za[k] = Azz[k + LAP][LAP][LAP];
        }
    }
    get_xs_blow_suction_multiwave(nx, nz, MZMAX, xa, za, SLZ, pfx, pgz, X_DIST_BEGIN, X_DIST_END);

    get_fait_multifrequancy(MTMAX);//Comput TM
    
    free(xa);
    free(za);

    memcpy_All(pub1, pub1_d->ptr, pub1_d->pitch, H2D, ny, 4, 1);
    memcpy_All(pfx, pfx_d->ptr, pfx_d->pitch, H2D, nx, 1, 1);
    memcpy_All(pgz, pgz_d->ptr, pgz_d->pitch, H2D, nz, 1, 1);

    
}

void get_xs_blow_suction_multiwave(int NX, int NZ, int MZ_MAX, REAL *xx,
REAL *zz, REAL SL, REAL *fx, REAL *gz, REAL DIST_BEGIN, REAL DIST_END){
    int MZ_MAX1;
    REAL ztmp, seta;
    REAL *faiz, *zl;
    
    MZ_MAX1 = abs(MZ_MAX);
    faiz = (REAL*)malloc(sizeof(REAL)*MZ_MAX1);
    zl = (REAL*)malloc(sizeof(REAL)*MZ_MAX1);
    
    ztmp = 0.;
    
    for(int k = 0; k < MZ_MAX1; k++){
        faiz[k] = rand()/(REAL)RAND_MAX;
        if(k == 0){
            zl[k] = 1.;
        }else{
            zl[k] = zl[k - 1] / 1.25;
        }
        ztmp = ztmp + zl[k];
    }
    
    for(int k = 0; k < MZ_MAX1; k++){
        zl[k] = zl[k] / ztmp;
    }
    
    MPI_Bcast(faiz, MZ_MAX1, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
    
    for(int i = 0; i < NX; i++){
        if(xx[i] >= DIST_BEGIN && xx[i] <= DIST_END){
            //seta = 2. * PI * (xx[i] - DIST_BEGIN)/(DIST_BEGIN - DIST_END);
            seta = 10. * PI * (xx[i] - DIST_BEGIN)/(DIST_BEGIN - DIST_END);
            fx[i] = 4. / sqrt(27.) * sin(seta) * (1. - cos(seta));
        }else{
            fx[i] = 0.;
        }
    }

    for(int k = 0; k < nz; k++){
        gz[k] = 0.;
        seta = zz[k] / SL;
        if(MZ_MAX > 0){
            for(int m = 0; m < MZ_MAX; m++){
                gz[k] = gz[k] + zl[m] * sin(2. * PI * (m + 1) * (seta + faiz[m]));
            }
        }else if(MZ_MAX == 0){
            gz[k] = 1.;
        }else{
            gz[k] = sin(-2. * PI * MZ_MAX * seta);
        }
    }

    free(faiz);
    free(zl);
}


void get_fait_multifrequancy(int MT_MAX){
    int Kflag = 0;
    REAL Ttmp = 0.;

    for(int k = 0; k < MT_MAX; k++){
        fait[k] = rand()/(REAL)RAND_MAX;
        TM[0] = 1.;
        if(k > 0) TM[k] = TM[k - 1] / 1.25;
        Ttmp = Ttmp + TM[k];
    }

    for(int k = 0; k < MT_MAX; k++){
        TM[k] = TM[k] / Ttmp;
    }

    MPI_Bcast(fait, MT_MAX, OCFD_DATA_TYPE, 0, MPI_COMM_WORLD);
}

#ifdef __cplusplus
}
#endif

#nonw

HOST_NAME=$(shell hostname)

SRC=$(wildcard src/*.c)
SRC+=$(wildcard src/*.cu)

HEAD=$(wildcard head/*.h)

OBJ=$(patsubst src/%c,obj/%o, $(SRC) )
OBJ:=$(patsubst src/%cu,obj/%o, $(OBJ) )


all: default
ifneq ($(shell which hipcc),)
# HIP compoler
#ifndef MPICH
#$(error env MPICH doesn't exist , MPI_PATH has wrong value)
#endif
#ifndef HIPCC
#$(error env HIP doesn't exist , DEV_PATH has wrong value)
#endif

#MPI_PATH=/opt/hpc/software/mpi/hpcx/v2.4.1/
MPI_PATH=/opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/
DEV_PATH=/opt/rocm/hip/


DEV=hipcc
HOST=mpicc

OPT_Commen=-O3

OPT_Host=-c -std=c99 -I $(DEV_PATH)/include -I $(DEV_PATH)/include/hip/hcc_detail/cuda -D __HIP_PLATFORM_HCC__ -D __HIPCC__
OPT_Host+= $(OPT_Commen)

OPT_Dev=-c -I /opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/include
OPT_Dev+=$(OPT_Commen)

SRC:=$(patsubst src/%.c,src_hip/%.c, $(SRC))
SRC:=$(patsubst src/%.cu,src_hip/%.cpp, $(SRC))

HEAD:=$(patsubst head/%.h,head_hip/%.h, $(HEAD))

.PRECIOUS : %.o %_hip.c %_hip.cpp %_hip.h

opencfd_hip.c : opencfd.c 
	hipify-perl $< > $@

opencfd.o : opencfd_hip.c $(HEAD)
	$(HOST) $(OPT_Host) -I head_hip/ -o opencfd.o opencfd_hip.c

hip_file : $(HEAD) $(SRC)

head_hip/%.h : head/%.h
	@if [ ! -e "head_hip" ] ; then mkdir head_hip ; fi
	hipify-perl $< > $@

src_hip/%.c : src/%.c 
	@if [ ! -e "src_hip" ] ; then mkdir src_hip ; fi
	hipify-perl $< > $@

src_hip/%.cpp : src/%.cu
	@if [ ! -e "src_hip" ] ; then mkdir src_hip ; fi
	hipify-perl $< > $@

src_hip/%.cpp : ana/%.c
	@if [ ! -e "src_hip" ] ; then mkdir src_hip ; fi
	hipify-perl $< > $@

src_hip/%.cpp : ana/%.cu
	@if [ ! -e "src_hip" ] ; then mkdir src_hip ; fi
	hipify-perl $< > $@

obj/%.o : src_hip/%.c head_hip/%.h
	@if [ ! -e "obj" ] ; then mkdir obj ; fi
	$(HOST) $(OPT_Host) -I head_hip/ $< -o $@

obj/%.o : src_hip/%.cpp head_hip/%.h
	@if [ ! -e "obj" ] ; then mkdir obj ; fi
	$(DEV) $(OPT_Dev) -I head_hip/ $< -o $@

clean:
	rm -rf *.o *.out obj/ src_hip/  head_hip/ opencfd_hip.c

else
#nvcc compiler

MPI_PATH=/usr/
#MPI_PATH=/home/dglin/intel/compilers_and_libraries_2019.4.243/linux/mpi/intel64/
DEV_PATH=/usr/local/cuda

#ifndef MPICH
#$(error env MPICH doesn't exist , MPI_PATH has wrong value)
#endif
#ifndef CUDA
#$(error env CUDA doesn't exist , DEV_PATH has wrong value)
#endif
#
#
#MPI_PATH=$(MPICH)
#DEV_PATH=$(CUDA)

DEV=nvcc
HOST=$(MPI_PATH)/bin/mpic++

OPT_Commen=-O3

OPT_Host=-c -std=c99 -I $(MPI_PATH)/include -I $(DEV_PATH)/include
OPT_Host+= $(OPT_Commen)

OPT_Dev=-dc -I $(MPI_PATH)/include -I $(DEV_PATH)/include
OPT_Dev+=$(OPT_Commen) -arch=compute_86

OpenCC_PATH=/home/sdp/SCU-comb/OpenCC


opencfd.o : opencfd.c
	$(HOST) $(OPT_Host) -I head/ -I $(OpenCC_PATH)/include -o opencfd.o opencfd.c

obj/%.o : src/%.c head/%.h
	@if [ ! -e "obj" ] ; then mkdir obj ; fi
	$(HOST) $(OPT_Host) -I head/ -I $(OpenCC_PATH)/include $< -o $@

obj/%.o : src/%.cu head/%.h
	@if [ ! -e "obj" ] ; then mkdir obj ; fi
	$(DEV) $(OPT_Dev) -I head/ -I $(OpenCC_PATH)/include $< -o $@


clean:
	rm -rf *.o *.out obj/

endif


default : opencfd.o obj/libocfd.a
	$(DEV) -O3 -o opencfd-scu.out opencfd.o -L obj -locfd -L $(MPI_PATH)/lib -L$(OpenCC_PATH)/lib -lmpi -lm -lpthread -lopencc


obj/libocfd.a : $(OBJ)
	ar -crv obj/libocfd.a $(OBJ)


ZIP_EXIST=0
zip : 
	@if [ -e "src_cuda.zip" ] ; then rm src_cuda.zip ; echo "rm src_cuda.zip"; fi
	zip --quiet -r src_cuda.zip head/ src/ test/ opencfd.c makefile README

echo:
	@echo $(HOST_NAME)
	@echo $(SRC)
	@echo $(OBJ)
	@echo $(HEAD)
	@echo $(A)

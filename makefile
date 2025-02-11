#nonw

HOST_NAME=$(shell hostname)

SRC=$(wildcard src/*.c)
SRC+=$(wildcard src/*.cu)

HEAD=$(wildcard head/*.h)

OBJ_DIR = obj
SRC_HIP_DIR = src_hip
HEAD_HIP_DIR = head_hip

OBJ=$(patsubst src/%c, $(OBJ_DIR)/%o, $(SRC))
OBJ:=$(patsubst src/%cu, $(OBJ_DIR)/%o, $(OBJ))
OBJ_C  = $(patsubst src/%c, $(OBJ_DIR)/%o, $(SRC))
OBJ_CU = $(patsubst src/%cu, $(OBJ_DIR)/%o, $(SRC))

all: default
ifneq ($(shell which hipcc),)

MPI_PATH=/opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/

#DEV=hipcc -fgpu-rdc
DEV=hipcc -fgpu-rdc
HOST=mpicc

OPT_Commen=-O3 

#DEV_PATH=/opt/rocm/hip/
#OPT_Host=-c -std=c99 -I $(DEV_PATH)/include -I $(DEV_PATH)/include/hip/hcc_detail/cuda -D __HIP_PLATFORM_HCC__ -D __HIPCC__
OPT_Host  = -c -std=c99 -D __HIP_PLATFORM_HCC__ -D __HIPCC__
OPT_Host += $(OPT_Commen)

#OPT_Dev=-c -I /opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/include
OPT_Dev  = -c 
OPT_Dev += $(OPT_Commen) 

SRC_HIP  = $(patsubst src/%.c,$(SRC_HIP_DIR)/%.c, $(SRC))
SRC_HIP :=$(patsubst src/%.cu,$(SRC_HIP_DIR)/%.cpp, $(SRC_HIP))

HEAD := $(patsubst head/%.h,$(HEAD_HIP_DIR)/%.h, $(HEAD))

.PRECIOUS : $(SRC_HIP)

opencfd.o : opencfd.c $(HEAD) 
	$(HOST) $(OPT_Host) -I $(HEAD_HIP_DIR)/ -o opencfd.o opencfd.c

$(HEAD_HIP_DIR)/%.h : head/%.h | $(HEAD_HIP_DIR)
	hipify-perl $< > $@

$(SRC_HIP_DIR)/%.c : src/%.c | $(SRC_HIP_DIR)
	hipify-perl $< > $@

$(SRC_HIP_DIR)/%.cpp : src/%.cu | $(SRC_HIP_DIR)
	hipify-perl $< > $@

$(OBJ_DIR)/%.o : $(SRC_HIP_DIR)/%.c $(HEAD_HIP_DIR)/%.h $(HEAD) 
	$(HOST) $(OPT_Host) -I $(HEAD_HIP_DIR)/ $< -o $@

$(OBJ_DIR)/%.o : $(SRC_HIP_DIR)/%.cpp $(HEAD_HIP_DIR)/%.h $(HEAD) 
	$(DEV) $(OPT_Dev) -I $(HEAD_HIP_DIR)/ $< -o $@

else
#nvcc compiler

MPI_PATH=/usr/local/mpi
#MPI_PATH=/home/dglin/intel/compilers_and_libraries_2019.4.243/linux/mpi/intel64/

DEV=nvcc
HOST=$(MPI_PATH)/bin/mpicc

OPT_Commen=-O3

#DEV_PATH=/usr/local/cuda
#OPT_Host=-c -std=c99 -I $(MPI_PATH)/include -I $(DEV_PATH)/include
OPT_Host=-c -std=c99 -I $(MPI_PATH)/include
OPT_Host+= $(OPT_Commen)

#OPT_Dev=-dc -I $(MPI_PATH)/include -I $(DEV_PATH)/include
OPT_Dev=-dc -I $(MPI_PATH)/include
OPT_Dev+=$(OPT_Commen) -code=sm_86 -arch=compute_86


opencfd.o : opencfd.c
	$(HOST) $(OPT_Host) -I head/ -o opencfd.o opencfd.c

$(OBJ_DIR)/%.o : src/%.c head/%.h | $(OBJ_DIR)
	$(HOST) $(OPT_Host) -I head/ $< -o $@

$(OBJ_DIR)/%.o : src/%.cu head/%.h | $(OBJ_DIR)
	$(DEV) $(OPT_Dev) -I head/ $< -o $@

endif

default : opencfd.o $(OBJ)
	$(DEV) -O3 -o opencfd-scu.out opencfd.o $(OBJ) -L $(MPI_PATH)/lib -lmpi -lm -lpthread


$(OBJ_DIR)/libocfd.a : $(OBJ)
	ar -crv $(OBJ_DIR)/libocfd.a $(OBJ)


$(OBJ_DIR) :
	mkdir -p $(OBJ_DIR)

$(SRC_HIP_DIR) :
	mkdir -p $(SRC_HIP_DIR)

$(HEAD_HIP_DIR) :
	mkdir -p $(HEAD_HIP_DIR)

ZIP_EXIST=0
zip : 
	@if [ -e "src_cuda.zip" ] ; then rm src_cuda.zip ; echo "rm src_cuda.zip"; fi
	zip --quiet -r src_cuda.zip head/ src/ test/ opencfd.c makefile README

echo:
	@echo $(HOST_NAME)
	@echo $(SRC)
	@echo $(SRC_HIP)
	@echo $(OBJ)
	@echo $(HEAD)

clean:
	rm -rf *.o *.out obj/* src_hip/* head_hip/*
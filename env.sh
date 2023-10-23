#!/bin/bash

module purge
module load compiler/devtoolset/7.3.1 
module load compiler/rocm/2.9
module load mpi/hpcx/2.4.1/gcc-7.3.1
module list

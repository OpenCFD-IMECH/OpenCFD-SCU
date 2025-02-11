#!/bin/bash

module purge

module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.4.1/gcc-7.3.1
module load python/3.8.10
module load compiler/rocm/dtk-23.04
module load compiler/cmake/3.23.3

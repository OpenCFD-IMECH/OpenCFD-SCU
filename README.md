### Program installation guide 
#### 1. Install CUDA (for NVIDIA environment) or ROCM (for AMD environment). Modify the installation path in the makefile - "DEV_PATH".

#### 2. Install MPI. Modify the installation path in the makefile - "MPI_PATH".

#### 3. The makefile will recognize the command line instruction "hipcc" to determine the environment.

#### 4. If you use NVIDIA environment, please modify "-code" and "-arch" according to the hardware architecture.

#### 5. Execute "make".





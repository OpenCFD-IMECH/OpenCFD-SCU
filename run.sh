#!/bin/bash                
#SBATCH -J name
#SBATCH -n 8
#SBATCH -N 2
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH -p blcy
#SBATCH --mem=64G

# Adding OMPI runtime parameters
export OMPI_MCA_btl_openib_allow_ib=true
export OMPI_MCA_pml=ucx

#default true
export SLURM_PMIX_DIRECT_CONN=true
export SLURM_PMIX_DIRECT_CONN_UCX=false

rm -f hostfile
for i in `scontrol show hostnames $SLURM_JOB_NODELIST`
do
 echo "$i slots=4" >> hostfile
done

mpirun -np $SLURM_NTASKS -hostfile ./hostfile --mca plm_rsh_no_tree_spawn 1 --mca plm_rsh_num_concurrent $SLURM_NTASKS -mca routed_radix $SLURM_NTASKS -mca pml ucx -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0 -x UCX_TLS=self,sm,dc -x UCX_DC_MLX5_TIMEOUT=5000ms -x LD_LIBRARY_PATH -mca btl_openib_warn_default_gid_prefix 0 -mca btl_openib_warn_no_device_params_found 0 --bind-to core ./single_process.sh ./opencfd-scu.out

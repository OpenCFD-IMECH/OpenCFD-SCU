#!/bin/bash

lrank=$OMPI_COMM_WORLD_LOCAL_RANK

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --membind=0 $@
  ;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --membind=1 $@
  ;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  numactl --membind=2 $@
  ;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  numactl --membind=3 $@
  ;;
esac
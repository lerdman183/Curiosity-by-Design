#!/bin/bash
# Code adopted from https://docs.alliancecan.ca/wiki/Deepspeed

source $SLURM_TMPDIR/env/bin/activate
export NCCL_ASYNC_ERROR_HANDLING=1

echo "r$SLURM_NODEID master: $HEAD_NODE"
echo "r$SLURM_NODEID Launching python script"

torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node 2 \ # This is the number of GPUs on each node!
--rdzv_backend c10d \
--rdzv_endpoint "$HEAD_NODE" \
Code/supplementary-material/clarification_module.py --deepspeed_config="./ds_config.json"
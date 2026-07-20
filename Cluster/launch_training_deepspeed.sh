#!/bin/bash
# Code adopted from https://docs.alliancecan.ca/wiki/Deepspeed

source $SLURM_TMPDIR/ENV/bin/activate

echo "Installing typing_extensions =================="
pip install --no-index typing_extensions
pip install --no-index typing-extensions

export NCCL_ASYNC_ERROR_HANDLING=1

echo "r$SLURM_NODEID master: $HEAD_NODE"
echo "r$SLURM_NODEID Launching python script"

torchrun \
--nnodes=$SLURM_NNODES \
--nproc_per_node=2 \
--rdzv_backend=c10d \
--rdzv_endpoint="$HEAD_NODE" \
Code/supplementary-material/clarification_module.py \
--deepspeed_config="./ds_config.json"
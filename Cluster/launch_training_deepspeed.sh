#!/bin/bash
# Code adopted from https://docs.alliancecan.ca/wiki/Deepspeed

module purge
module load python gcc arrow/21.0.0 cuda

cd projects/def-lutellie/Curiosity-by-Design

source venv/bin/activate

export NCCL_ASYNC_ERROR_HANDLING=1
# Force NCCL to use the correct network interface if needed (Alliance cluster standard)
export NCCL_IB_DISABLE=0

echo "r$SLURM_NODEID master: $HEAD_NODE"
echo "r$SLURM_NODEID Launching python script"

torchrun \
--nnodes=$SLURM_NNODES \
--nproc_per_node=2 \
--rdzv_backend=c10d \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_endpoint="$HEAD_NODE:$MASTER_PORT" \
Code/supplementary-material/clarification_module.py \
--deepspeed_config="./ds_config.json"
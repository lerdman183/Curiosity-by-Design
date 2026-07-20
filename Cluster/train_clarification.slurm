#!/bin/bash
# Code adopted from https://docs.alliancecan.ca/wiki/Deepspeed
#SBATCH --job-name=clarify-gemma
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=18000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd ~/Curiosity-by-Design

# Create a virtualenv and install deepspeed + its dependencies on all nodes
srun -N $SLURM_NNODES -n $SLURM_NNODES Cluster/config_env.sh

export HEAD_NODE=$(hostname) # store head node's address

module load cuda

srun Cluster/launch_training_deepspeed.sh
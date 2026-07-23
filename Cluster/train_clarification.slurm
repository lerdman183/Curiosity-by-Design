#!/bin/bash
#SBATCH --job-name=clarify-llama
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/Curiosity-by-Design/logs/%x-%j.out
#SBATCH --error=/scratch/%u/Curiosity-by-Design/logs/%x-%j.err

# Before submitting this file, run the following lines in the command prompt to set up the directories, virtual environment and install packages:
# ===============================
# Set up output directories (replace lerdman with your username, only have to do once)
# mkdir -p ~/scratch/lerdman/Curiosity-by-Design/logs
# mkdir -p ~/scratch/lerdman/Curiosity-by-Design/llama3.1-8B-ft-clarification
#
# cd ~/projects/def-lutellie/Curiosity-by-Design
# module purge
# module load python gcc arrow/21.0.0 cuda
# virtualenv --no-download venv
# source venv/bin/activate
# pip install --upgrade pip --no-index
# pip install --no-index torch transformers peft datasets wandb huggingface_hub deepspeed typing_extensions bitsandbytes pathlib
# ===============================



cd ~/projects/def-lutellie/Curiosity-by-Design

export HEAD_NODE=$(hostname)
export MASTER_PORT=40000

srun --ntasks=$SLURM_NNODES Cluster/launch_training_deepspeed.sh
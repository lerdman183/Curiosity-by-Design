#!/bin/bash
#SBATCH --job-name=clarify-llama
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Before submitting this file, run the following lines in the command prompt to set up the virtual environment and install packages:\
# ===============================
# cd ~/Curiosity-by-Design
# module purge
# module load python gcc arrow/21.0.0 cuda
# virtualenv --no-download venv
# source venv/bin/activate
# pip install --upgrade pip --no-index
# pip install --no-index torch transformers peft datasets wandb huggingface_hub deepspeed typing_extensions
# ===============================

cd ~/Curiosity-by-Design

export HEAD_NODE=$(hostname) # store head node's address

srun Cluster/launch_training_deepspeed.sh
#!/bin/bash
#SBATCH --job-name=clarify-gemma
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge

module load python gcc arrow/21.0.0

source ~/venv/bin/activate

cd ~/Curiosity-by-Design

python Code/supplementary-material/clarification_module.py
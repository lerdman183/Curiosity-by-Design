#!/bin/bash
#SBATCH --job-name=clarify-gemma
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge

module load python
module load cuda

pip install torch torchvision torchaudio transformers peft datasets scikit-learn wandb huggingface_hub jupyter pandas --index-url https://download.pytorch.org/whl/cu128

source ~/venv/bin/activate

cd ~/Curiosity-by-Design

python Code/supplementary-material/clarification_module.py
#!/bin/bash
#SBATCH --job-name=test-clarify-llama-2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80000M
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/Curiosity-by-Design/logs/%x-%j.out
#SBATCH --error=/scratch/%u/Curiosity-by-Design/logs/%x-%j.err

# Before running this file, run the following lines in the terminal to set up the directories, virtual environment and install packages:
# ===========================================================================
# mkdir -p ~/scratch/Curiosity-by-Design/results
#
# cd ~/projects/def-lutellie/Curiosity-by-Design
# module purge
# module load python gcc arrow/21.0.0 cuda
# virtualenv --no-download venv
# source venv/bin/activate
# pip install --upgrade pip --no-index
# pip install --no-index torch transformers peft huggingface_hub typing_extensions bitsandbytes pathlib sklearn
# hf auth login
# ===========================================================================

cd ~/projects/def-lutellie/Curiosity-by-Design

python Code/supplementary-material/test_clarification_module.py
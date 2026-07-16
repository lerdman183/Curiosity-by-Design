#!/bin/bash
#SBATCH --job-name=clarify-gemma
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=18000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge

module load python

virtualenv --no-download $SLURM_TMPDIR/ENV

module load python gcc arrow/21.0.0

source $SLURM_TMPDIR/ENV/bin/activate

pip install --upgrade pip --no-index

pip install --no-index torch torchvision torchaudio transformers peft datasets scikit-learn wandb huggingface_hub jupyter pandas

python -c "import torch; print(torch.__file__)"

python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"

ls -l $(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

python -c "import torch; print(torch.__version__); print(torch.__file__)"

pip show torch torchaudio

echo "Done installing virtualenv!"

srun python Code/supplementary-material/clarification_module.py
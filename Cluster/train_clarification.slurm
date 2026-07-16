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

echo "Python:"
which python

echo
echo "Pip:"
pip --version

echo
echo "Torch:"
python -c "import torch; print(torch.__version__)"

echo
echo "Torchaudio:"
python -c "import torchaudio; print(torchaudio.__version__)"

echo
echo "Shared libraries:"
ldd $VIRTUAL_ENV/lib/python3.11/site-packages/torchaudio/lib/libtorchaudio.so

echo "Done installing virtualenv!"

srun python Code/supplementary-material/clarification_module.py
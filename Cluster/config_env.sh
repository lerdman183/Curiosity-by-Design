#!/bin/bash
# Code adapted from https://docs.alliancecan.ca/wiki/Deepspeed

module purge

module load python

virtualenv --no-download $SLURM_TMPDIR/ENV

module load gcc arrow/21.0.0

source $SLURM_TMPDIR/ENV/bin/activate

pip install --upgrade pip --no-index

pip install --no-index typing_extensions

pip install --no-index torch transformers peft datasets wandb huggingface_hub deepspeed

echo "Python:"
which python

echo "Pip:"
which pip

echo "Torchrun:"
which torchrun

python -m pip show typing_extensions
python -m pip show torch

python -c "import typing_extensions; print('typing_extensions OK')"
python -c "import torch; print(torch.__version__)"

echo "Done installing virtualenv!"
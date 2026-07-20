#!/bin/bash
# Code adapted from https://docs.alliancecan.ca/wiki/Deepspeed

module purge

module load python

virtualenv --no-download $SLURM_TMPDIR/ENV

module load gcc arrow/21.0.0

source $SLURM_TMPDIR/ENV/bin/activate

pip install --upgrade pip --no-index

pip install --no-index torch transformers peft datasets wandb huggingface_hub deepspeed typing_extensions

echo "Done installing virtualenv!"
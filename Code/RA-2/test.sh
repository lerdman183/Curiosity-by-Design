#!/bin/bash

# Log in to Hugging Face without manual input
echo "Logging into Hugging Face..."
huggingface-cli login ***REMOVED-HF-TOKEN***

echo "HF login success! Logging into Wandb."
# Log in to Wandb
wandb login ***REMOVED-WANDB-TOKEN***
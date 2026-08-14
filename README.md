# Acknowlegements
This project uses Meta Llama 3.3 70B Instruct as the base model for the clarification question generation system. We gratefully acknowledge Meta for developing and releasing the Llama family of large language models. The model was accessed through Hugging Face and used in accordance with the applicable Llama 3.3 Community License and usage policies.

Base model: Meta Llama 3.3 70B Instruct
Developer: Meta AI

Use of this model is subject to the Llama 3.3 Community License and Meta's applicable usage policies.

This project is not affiliated with, sponsored by, or endorsed by Meta.

---

This project uses DeBERTa-v3-base, developed by Microsoft, as part of the clarification question generation pipeline. We gratefully acknowledge the authors for developing the DeBERTa architecture and making the model publicly available.

Model: Microsoft DeBERTa-v3-base
License: MIT

The DeBERTa-v3-base model is provided under the MIT License. This project is not affiliated with, sponsored by, or endorsed by Microsoft.


# Curiosity by Design

**An LLM-based coding assistant that asks clarification questions.**

This project explores how a coding assistant can detect when a user's prompt is vague or under-specified, ask a clarification question, and then use the user's reply to produce a better final answer. It combines two trained models into a pipeline:

1. **Intent Classifier** — a deBERTa_v3_base model that scores how ambiguous a prompt is, on a scale of 1 to 4.
2. **Clarification Module** — a fine-tuned Meta Llama-3.3-70B-Instruct model that asks a follow-up question when the prompt is flagged as ambiguous.

---

## What's in this repo

| Folder | What it contains |
|--------|------------------|
| `Cluster/` | The job scripts used to run the training/testing scripts on a cluster. |
| `Code/supplementary-material/` | The main pipeline: training scripts, testing scripts, and the end-to-end pipeline test. Start here. |
| `Code/RA-2/` | Earlier prototype — scripts for mining GitHub PR comments, categorizing them, and an initial classifier experiment. |
| `Code/revised-ra2-iclr/` | A revised version of the RA-2 work prepared for ICLR. |
| `Code/Leveraging .../` | Reproduction of prior work on conversational data for ambiguous review suggestions. |
| `Datasets/` | Cleaned datasets used for training and evaluation (categorized PR comments, synthetic prompts, etc.). |
| `Notebooks/` | Jupyter notebooks for data synthesis (`dataset-synth.ipynb`) and exploratory analysis (`research_project_2025.ipynb`). |
| `Papers/` | Reference papers cited throughout the project. |

---

## Results
If reproductability is not required, recent results can be found under `Datasets/results/`. For older results/user tests, see `Code/Gemma-work/` and `Datasets/unused-old-data/` 

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/lerdman183/Curiosity-by-Design.git
cd Curiosity-by-Design
```

### 2. Set up a Python environment

A virtual environment keeps this project's dependencies separate from the rest of your system.

```bash
python -m venv venv
source venv/bin/activate           # on macOS / Linux
# .\venv\Scripts\activate          # on Windows
```

### 3. Install dependencies

```bash
pip install torch transformers peft bitandbytes wandb huggingface_hub scikit-learn pathlib sklearn
```

> Exact versions are not pinned.

### 4. Log in to Hugging Face and Weights & Biases

The training scripts download base models from Hugging Face and log metrics to Weights & Biases. Both require accounts.

```bash
huggingface-cli login          # paste your HF token when prompted
wandb login                    # paste your W&B API key when prompted
```

Get tokens from:
- Hugging Face: https://huggingface.co/settings/tokens
- Weights & Biases: https://wandb.ai/authorize

**Do not commit these tokens.** Use environment variables or the interactive login above.

---

## Running the pipeline

All training and testing scripts live under `Code/supplementary-material/` and `Cluster/`.

To train the Llama 3.3 70B Instruct model, access to a cluster is needed. Without cluster use, memory requirements will not be met

### Train the Intent Classifier

```bash
python Code/supplementary-material/intent_classifier.py
```

### Train the Clarification Module - NEED CLUSTER
Follow pre-run steps commented in train_clarification.slurm

To start the training run:
```bash
sbatch Cluster/train_clarification.slurm
```

### Test the Intent Classifier

```bash
python Code/supplementary-material/test_classifier.py
```

### Test the Clarification Module - NEED CLUSTER
Follow pre-run steps commented in test_clarification.slurm

To start the testing run:
```bash
sbatch Cluster/test_clarification.slurm
```

---

## Hardware notes

Fine-tuning the deBERTa-v3-base model requires a CUDA equipped GPU to run. The deBERTa-v3-large model could potentially be trained with larger GPU sizes or a LoRA/QLoRA adapter.

Fine-tuning the Llama-3.3-70B-Instruct model requires the use of h100 GPUs on a cluster. If these are not used, memory requirements will not be met and the model will not be able to be loaded/trained properly. The cluster used in this project was Nibi, which was capable of allocating 4 h100 GPUs in the same cluster. If this is not possible, h100 GPUs may be split across multiple nodes, although this may take longer for the job to be scheduled, and the script will run slower. To change this so nodes can be used across multiple nodes, make changes in `Cluster/train_clarification.slurm` and `Cluster/launch_training_deepspeed.sh`. For example, for two h100s across two nodes change the #SBATCH --gres=gpu:h100:4 to #SBATCH --gres=gpu:h100:2, add #SBATCH --nodes=2, and change --nproc_per_node=4 to --nproc_per_node=2

Testing the Llama-3.3-70B-Instruct model also requires the use of an h100 GPU to load the weights into memory. This also requires the use of a cluster, though only one h100 must be present in a given node.

 This project made us of the alliancecan clusters. For more information, see https://docs.alliancecan.ca/wiki/Technical_documentation

---

## Project structure (high level)

```
Curiosity by Design/
├── Cluster/
├── Code/
│   ├── Gemma-work/               <- earlier work with Gemma
│   ├── supplementary-material/   <- start here
│   ├── RA-2/                     <- earlier prototype
│   ├── revised-ra2-iclr/         <- ICLR-revised version
│   └── Leveraging .../           <- prior-work reproduction
├── Datasets/
├── Notebooks/
└── Papers/
```

---

## Troubleshooting

- **`huggingface-cli: command not found`** — Install with `pip install huggingface_hub`.
- **CUDA out of memory** — Lower the batch size in the training script, or run on CPU (slow and only able to do for intent classifier)
- **`ModuleNotFoundError`** — Make sure your virtual environment is activated and all dependencies are installed.

For troubleshooting issues on a cluster, see `Cluster/cluster_common_problems.pdf`
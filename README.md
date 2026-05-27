# Curiosity by Design

**An LLM-based coding assistant that asks clarification questions.**

This project explores how a coding assistant can detect when a user's prompt is vague or under-specified, ask a clarification question, and then use the user's reply to produce a better final answer. It combines two trained models into a pipeline:

1. **Intent Classifier** — a DistilBERT model that scores how ambiguous a prompt is, on a scale of 1 to 4.
2. **Clarification Module** — a fine-tuned Gemma-3-1b-it model that asks a follow-up question when the prompt is flagged as ambiguous.

A baseline Gemma-3-1b-it model then generates the final answer, enriched with the user's clarification.

---

## What's in this repo

| Folder | What it contains |
|--------|------------------|
| `Code/supplementary-material/` | The main pipeline: training scripts, testing scripts, and the end-to-end pipeline test. Start here. |
| `Code/RA-2/` | Earlier prototype — scripts for mining GitHub PR comments, categorizing them, and an initial classifier experiment. |
| `Code/revised-ra2-iclr/` | A revised version of the RA-2 work prepared for ICLR. |
| `Code/Leveraging .../` | Reproduction of prior work on conversational data for ambiguous review suggestions. |
| `Datasets/` | Cleaned datasets used for training and evaluation (categorized PR comments, synthetic prompts, etc.). |
| `Notebooks/` | Jupyter notebooks for data synthesis (`dataset-synth.ipynb`) and exploratory analysis (`research_project_2025.ipynb`). |
| `Papers/` | Reference papers cited throughout the project. |
| `User-Study/` | User study materials and (anonymized) participant responses for RQ1 and RQ2. |
| `Models/` | Local landing folder for model weights. **Not tracked in git** — download from Zenodo (see below). |
| `zenodo/` | Local landing folder for the published supplementary bundle. **Not tracked in git** — download from Zenodo (see below). |

---

## Where to get the data and model weights

The model checkpoints and large dataset bundles are hosted on Zenodo (too large for git):

[Supplementary material on Zenodo](https://zenodo.org/records/15493099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImYwYWQwZGFkLTUzMjMtNGU3Yi05YzYyLWVmNTIzNzM4ZTAyYiIsImRhdGEiOnt9LCJyYW5kb20iOiJhOTA4MWUzZDk4ZmY5Nzg3NmQxM2VkMjk1ZWNlNGRlZiJ9.D4SggWOEQOuXVDpy7FoPFUTFO4YZizCXeKDVywEbuRT6wJkadFKk9E_gOTlNGsxRY6QVfK94bott79KdCRy1JQ)

You'll find:
- `checkpoint-8424.zip` — Intent Classifier weights
- `gemma3-1b-it-ft-new-data.zip` — Clarification Module weights
- `clarification_module_synth_dataset.json` — synthetic dataset for the Clarification Module
- `classifier_train_dataset.json` — labeled prompts for the Intent Classifier
- `test_pipeline.csv` — end-to-end test cases

After downloading, unzip the checkpoints into the `Models/` folder.

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/JustHarsh/Curiosity-by-Design.git
cd "Curiosity-by-Design"
```

### 2. Set up a Python environment

A virtual environment keeps this project's dependencies separate from the rest of your system.

```bash
python3 -m venv venv
source venv/bin/activate          # on macOS / Linux
# .\venv\Scripts\activate          # on Windows
```

### 3. Install dependencies

```bash
pip install torch transformers peft datasets scikit-learn wandb huggingface_hub jupyter pandas
```

> Exact versions are not pinned. If you need reproducibility, you'll want to capture working versions with `pip freeze > requirements.txt`.

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

### 5. Download the data and weights

Visit the Zenodo link above, download the bundle, and unzip the checkpoints into `Models/`.

---

## Running the pipeline

All training and testing scripts live under `Code/supplementary-material/`.

### Train the Intent Classifier

```bash
python Code/supplementary-material/intent_classifier.py
```

### Train the Clarification Module

```bash
python Code/supplementary-material/clarification_module.py
```

### Test the Intent Classifier

```bash
python Code/supplementary-material/test_classifier.py
```

### Test the Clarification Module

```bash
python Code/supplementary-material/test_clarification_module.py
```

### Run the end-to-end pipeline test

This combines the Intent Classifier, Clarification Module, and baseline Gemma-3-1b-it model on the prompts in `test_pipeline.csv`.

```bash
python Code/supplementary-material/test_pipeline.py
```

---

## User study

The `User-Study/` folder contains the materials used for the two research questions:

- **RQ1** — Are the clarification questions themselves useful? Annotators rate clarification questions from the Clarification Module against those from baseline Gemma-3-1b-it on Precision and Focus, Immediate Editability, and Contextual Fit.
- **RQ2** — Does the full pipeline produce better final answers? Annotators rate full pipeline responses against the baseline on Precision and Focus, Contextual Fit, Answer Faithfulness, and Correctness.

Sample annotation documents and anonymized participant responses are included.

---

## Hardware notes

Fine-tuning Gemma-3-1b-it benefits from a GPU. The Clarification Module training uses LoRA (parameter-efficient fine-tuning), which keeps GPU memory requirements manageable, but a CUDA-capable GPU is still strongly recommended. Inference and the Intent Classifier are lighter and can run on CPU if you're patient.

---

## Project structure (high level)

```
Curiosity by Design/
├── Code/
│   ├── supplementary-material/   <- start here
│   ├── RA-2/                     <- earlier prototype
│   ├── revised-ra2-iclr/         <- ICLR-revised version
│   └── Leveraging .../           <- prior-work reproduction
├── Datasets/
├── Notebooks/
├── Papers/
├── User-Study/
├── Models/   (gitignored — populate from Zenodo)
└── zenodo/   (gitignored — populate from Zenodo)
```

---

## Troubleshooting

- **`huggingface-cli: command not found`** — Install with `pip install huggingface_hub`.
- **CUDA out of memory** — Lower the batch size in the training script, or run on CPU (slow).
- **Missing model weights** — Download from Zenodo and unzip into `Models/`.
- **`ModuleNotFoundError`** — Make sure your virtual environment is activated and all dependencies are installed.

import json
import os
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DefaultDataCollator,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import math


def load_and_preprocess(json_filename):
    examples = []
    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        prompt = entry.get("prompt", "").strip()
        question = entry.get("clarifying_question", "").strip()
        if prompt and question:
            examples.append({
                "input": prompt,
                "output": question
            })
    return examples

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_filename = os.path.join(BASE_DIR, "..", "..", "Datasets", "gerrit-cleaning", "filtered_questions.json")
text_examples = load_and_preprocess(json_filename)

# 70/30 train/eval split
# Random state is fixed so the split is reproducible and so test_clarification_module.py can regenerate/verify if needed
train_examples, eval_examples = train_test_split(
    text_examples, test_size=0.3, random_state=42
)

class TextDataset(TorchDataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text = ex["input"].strip()
        answer_text = ex["output"].strip()

        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": answer_text},
        ]

        # Get plain python lists of token ids (no tensor conversion here)
        full_ids_list = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_only_ids_list = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=True,
        )

        # Convert to tensors ourselves — avoids the return_tensors="pt" bug
        input_ids = torch.tensor(full_ids_list, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[: len(prompt_only_ids_list)] = -100

        # Pad/truncate to max_length
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            labels = labels[-self.max_length:]
        else:
            pad_len = self.max_length - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# Switch the cache directory to scratch as to not exceed quota in the home directory
CACHE_DIR = os.path.join(os.environ["SCRATCH"], "Curiosity-by-Design", "hf_cache")

model_checkpoint = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = TextDataset(train_examples, tokenizer, max_length=512)
eval_dataset = TextDataset(eval_examples, tokenizer, max_length=512)

# Ensure these numbers match those in ds_config.json
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 5e-5
EPOCHS = 5

# -----------------------------
# 3. WandB init (unchanged)
# -----------------------------
IS_RANK_ZERO = int(os.environ.get("RANK", "0")) == 0
if __name__ == "__main__" and IS_RANK_ZERO: # only initialize wandb when running as main script on the rank 0
    wandb.init(
        project="Curiosity-By-Design",
        name="llama3.3-70B-ft-clarification-gerrit",
        mode="offline", # set to "offline" for running on cluster
        config={
            "model": model_checkpoint,
            "dataset": json_filename,
            "train_size": len(train_examples),
            "eval_size": len(eval_examples),
            "epochs": EPOCHS,
            "seq_len": 512,
            "batch_per_device": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LR,
        },
    )

# Write the eval split to disk once
EVAL_SPLIT_PATH = os.path.join(BASE_DIR, "..", "..", "Datasets", "clarification_module_eval_split.json")

if __name__ == "__main__" and IS_RANK_ZERO:
    with open(EVAL_SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_examples, f, indent=2)

# -----------------------------
# 4. Model + QLoRA on multiple GPUs 
# -----------------------------
assert torch.cuda.is_available(), "CUDA is required for this script"

# Set up QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    #bnb_4bit_use_double_quant=True,
)

# Attach LoRA adapter
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Adjusted for LLaMA-3
)

# Use the local rank to map processes across multiple GPUs
local_rank = int(os.environ.get("LOCAL_RANK", 0))

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    attn_implementation="eager",   # explicit eager attention
    use_cache=False,               # required for checkpointing
    device_map={"": local_rank},
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)

# -----------------------------
# 5. Trainer
# -----------------------------
data_collator = DefaultDataCollator()

class PerplexityLoggingCallback(TrainerCallback):
    """Derives eval perplexity (exp(eval_loss)) each time Trainer evaluates,
    and logs it to wandb alongside the automatically logged eval_loss."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["eval_perplexity"] = perplexity
        if IS_RANK_ZERO and wandb.run is not None:
            wandb.log({"eval/perplexity": perplexity}, step=state.global_step)
        print(f"Step {state.global_step}: eval_loss={metrics['eval_loss']:.4f}, perplexity={perplexity:.4f}")

DS_CONFIG_PATH = os.path.join(BASE_DIR, "..", "..", "Cluster", "ds_config.json")
training_args = TrainingArguments(
    output_dir=os.path.join(os.environ["SCRATCH"], "Curiosity-by-Design", "llama3.3-70B-ft-clarification-gerrit"),

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,

    logging_steps=10,
    eval_strategy="epoch",   # Run loss-based eval each epoch
    save_strategy="epoch",
    save_total_limit=3,

    bf16=True,                          # mixed precision
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True}, # Needed for gradient checkpointing
    report_to="wandb",
    run_name="llama3.3-70B-ft-clarification-gerrit",
    dataloader_num_workers=1,
    deepspeed=DS_CONFIG_PATH,  # Use DeepSpeed for parallel training
)

# Don't start training if the script is run by a worker
if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[PerplexityLoggingCallback()],
    )

    # -----------------------------
    # 6. Train & save
    # -----------------------------
    # Resume from the last checkpoint if there is one
    output_path = Path(training_args.output_dir)
    checkpoints = list(output_path.glob("checkpoint-*")) if output_path.exists() else []
    last_checkpoint = str(max(checkpoints, key=lambda p: p.stat().st_mtime)) if checkpoints else None

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # -----------------------------
    # 7. Loss-based eval on the held-out 30%
    # -----------------------------
    metrics = trainer.evaluate()
    print("Final eval metrics (loss/perplexity, not question-rate):", metrics)
    print(f"Eval split saved to {EVAL_SPLIT_PATH} for use by test_clarification_module.py")
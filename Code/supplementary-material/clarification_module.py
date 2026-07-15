import json
import os
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb


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
json_filename = os.path.join(BASE_DIR, "..", "..", "Datasets", "clarification_module_synth_dataset.json")
text_examples = load_and_preprocess(json_filename)

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

        # Append EOS so model learns to end generation
        prompt = prompt_text + self.tokenizer.eos_token
        answer = answer_text + self.tokenizer.eos_token

        # Tokenize prompt and answer separately
        prompt_tok = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        answer_tok = self.tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors="pt"
        )

        # Concatenate prompt + answer
        input_ids = torch.cat([prompt_tok.input_ids[0], answer_tok.input_ids[0]], dim=0)
        attention_mask = torch.cat([prompt_tok.attention_mask[0], answer_tok.attention_mask[0]], dim=0)

        # Create labels that mask out the prompt portion
        labels = input_ids.clone()
        labels[: prompt_tok.input_ids.size(1) ] = -100   # no loss on prompt

        # Pad/truncate to max_length
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]
            labels = labels[-self.max_length :]
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


model_checkpoint = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = TextDataset(text_examples, tokenizer, max_length=512)

# -----------------------------
# 3. WandB init (unchanged)
# -----------------------------
if __name__ == "__main__": # only initialize wandb when running as main script
    wandb.init(
        project="Curiosity-By-Design",
        name="gemma3-1b-it-singleGPU-cc",
        mode="offline", # set to "offline" for running on cluster
        config={
            "model": model_checkpoint,
            "dataset": json_filename,
            "epochs": 3,
            "seq_len": 512,
            "batch_per_device": 1,
            "grad_accum": 2,
            "learning_rate": 5e-5,
        },
    )

# -----------------------------
# 4. Model + LoRA on single GPU
# -----------------------------
assert torch.cuda.is_available(), "CUDA is required for this script"
torch.cuda.set_device(0)

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    dtype=torch.float16,
    device_map={"": 0},  # everything on cuda:0
    attn_implementation="eager",   # explicit eager attention
    use_cache=False,               # required for checkpointing
)

# Attach LoRA adapter
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj"],
)

# enable gradient checkpointing first
model.gradient_checkpointing_enable()

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing for activations
#model.gradient_checkpointing_enable()

# -----------------------------
# 5. Trainer
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gemma3-1b-it-ft-new-data",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,      # batch size = 1
    gradient_accumulation_steps=2,      # effective batch = 2
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    fp16=True,                          # mixed precision
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="gemma3-1b-it-singleGPU",
    dataloader_num_workers=2,
)

# Don't start training if the script is run by a worker
if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # -----------------------------
    # 6. Train & save
    # -----------------------------
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
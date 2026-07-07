import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom Dataset
class ClarificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Set project name for wandb
os.environ["WANDB_PROJECT"] = "Curiosity-By-Design"

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "..", "Datasets", "classifier_train_dataset.json")

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter out incomplete entries
data = [item for item in data if "prompt" in item and "clarification_need" in item and isinstance(item["clarification_need"], int)]

texts = [item["prompt"] for item in data]
labels = [int(item["clarification_need"]) - 1 for item in data]  # 0-based

# Train/validation split (80/20)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

#TODO: Consider using a larger model like "microsoft/deberta-v3-large" for better performance, but be aware of increased resource requirements.
MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False) # Using use_fast=False to avoid potential issues with tokenization
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

train_dataset = ClarificationDataset(train_texts, train_labels, tokenizer)
val_dataset = ClarificationDataset(val_texts, val_labels, tokenizer)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Run on GPU if available
assert torch.cuda.is_available(), "CUDA is required for this script"

training_args = TrainingArguments(
    output_dir="./results_scratch",

    num_train_epochs=10,
    weight_decay=0.01,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    learning_rate=2e-5,

    logging_dir="./logs_scratch",
    logging_steps=20,

    load_best_model_at_end=True,
    greater_is_better=True,
    metric_for_best_model="f1",
    seed=42,
    bf16=True,  # Enable mixed precision training for faster training and lower memory usage

    report_to="wandb",
    run_name="intent_classifier_deberta_v3_base"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Accuracy:  {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall:    {eval_results['eval_recall']:.4f}")
print(f"F1 Score:  {eval_results['eval_f1']:.4f}")
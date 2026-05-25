import os
import json
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom Dataset
class ClarificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
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

# Main
os.environ["WANDB_DISABLED"] = "true"
dataset_path = "/content/coding_prompts_dataset.json"

# Load dataset
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

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

train_dataset = ClarificationDataset(train_texts, train_labels, tokenizer)
val_dataset = ClarificationDataset(val_texts, val_labels, tokenizer)

training_args = TrainingArguments(
    output_dir="./results_scratch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_scratch",
    logging_steps=10,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Accuracy:  {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall:    {eval_results['eval_recall']:.4f}")
print(f"F1 Score:  {eval_results['eval_f1']:.4f}")
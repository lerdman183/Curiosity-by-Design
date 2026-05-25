import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import Dataset

# Set device to MPS if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load correct datasets
train_file = 'train.tsv'
test_file = 'test_with_labels.tsv'
train_df = pd.read_csv(train_file, sep='\t', dtype=str).dropna(subset=['initial_request', 'clarification_need'])
test_df = pd.read_csv(test_file, sep='\t', dtype=str).dropna(subset=['initial_request', 'clarification_need'])

# Convert to lists
train_texts, train_labels = train_df['initial_request'].tolist(), train_df['clarification_need'].astype(int).tolist()
test_texts, test_labels = test_df['initial_request'].tolist(), test_df['clarification_need'].astype(int).tolist()

# Define tokenizer and model
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize inputs
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# Define dataset class
class ClarificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long tensors
        return item

train_dataset = ClarificationDataset(train_encodings, train_labels)
test_dataset = ClarificationDataset(test_encodings, test_labels)

import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 4)  # 4-class classification

    def forward(self, input_ids, attention_mask, labels=None):  # Accept labels for loss computation
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # CLS token output

        if labels is not None:
            loss = F.cross_entropy(logits, labels)  # Compute loss inside the model
            return loss, logits
        return logits


# Train model with fine-tuning
model = TransformerClassifier(MODEL_NAME).to(device)  # Move model to MPS

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate fine-tuned model
results_finetuned = trainer.evaluate()
print("Fine-tuned Model Performance:", results_finetuned)

# Make predictions on the test set
def predict_on_test(texts):
    encodings = tokenize_function(texts)
    input_ids = encodings['input_ids'].to(device)  # Move input tensors to MPS
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Move back to CPU
    return preds

test_preds = predict_on_test(test_texts)

# Save predictions to file
test_df['predicted_clarification_need'] = test_preds
test_df.to_csv('fine_tuned_test_predictions.tsv', sep='\t', index=False)
print("Predictions saved to fine_tuned_test_predictions.tsv")

# Generate binary confusion matrix
binary_test_labels = [0 if lbl in [1, 2] else 1 for lbl in test_labels]
binary_test_preds = [0 if pred in [1, 2] else 1 for pred in test_preds]

conf_matrix = confusion_matrix(binary_test_labels, binary_test_preds)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print("Binary Confusion Matrix:")
print(conf_matrix_df)

# Compute evaluation metrics
acc = accuracy_score(binary_test_labels, binary_test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(binary_test_labels, binary_test_preds, average='binary')

print("Fine-Tuned Model Binary Classification Performance:")
print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

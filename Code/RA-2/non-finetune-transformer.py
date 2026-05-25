import pandas as pd
from transformers import pipeline

# Load test dataset
test_file = 'test_with_labels.tsv'
test_df = pd.read_csv(test_file, sep='\t', dtype=str).dropna(subset=['initial_request', 'clarification_need'])

test_texts = test_df['initial_request'].tolist()
test_labels = test_df['clarification_need'].astype(int).tolist()

# Load zero-shot classification model
MODEL_NAME = 'facebook/bart-large-mnli'
classifier = pipeline("zero-shot-classification", model=MODEL_NAME)

# Define classification labels
labels = ["No Clarification Needed", "Clarification Needed"]

# Run zero-shot classification
preds = [classifier(text, candidate_labels=labels)['labels'][0] for text in test_texts]

# Convert predictions back to binary format
predicted_labels = [0 if pred == "No Clarification Needed" else 1 for pred in preds]

# Save predictions to file
test_df['predicted_clarification_need'] = predicted_labels
test_df.to_csv('zero_shot_test_predictions.tsv', sep='\t', index=False)
print("Predictions saved to zero_shot_test_predictions.tsv")

# Generate binary confusion matrix
binary_test_labels = [0 if lbl in [1, 2] else 1 for lbl in test_labels]

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

conf_matrix = confusion_matrix(binary_test_labels, predicted_labels)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print("Binary Confusion Matrix:")
print(conf_matrix_df)

# Compute evaluation metrics
acc = accuracy_score(binary_test_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(binary_test_labels, predicted_labels, average='binary')

print("Zero-Shot Classification Model Performance:")
print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

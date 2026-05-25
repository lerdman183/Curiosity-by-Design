import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
v3_df = pd.read_csv('V3_predicted_clarification_need_from_scratch.tsv', sep='\t', dtype=str)

# Ensure the fields exist
if 'predicted_clarification_need' not in v3_df.columns or 'clarification_need' not in v3_df.columns:
    raise ValueError("Columns 'predicted_clarification_need' and 'clarification_need' must be present in the file.")

# Convert values to integers if needed
v3_df = v3_df.dropna(subset=['predicted_clarification_need', 'clarification_need'])
v3_df['predicted_clarification_need'] = v3_df['predicted_clarification_need'].astype(int)
v3_df['clarification_need'] = v3_df['clarification_need'].astype(int)

# Convert to binary classification: (1,2) -> 0 and (3,4) -> 1
v3_df['binary_predicted'] = v3_df['predicted_clarification_need'].apply(lambda x: 0 if x in [1, 2] else 1)
v3_df['binary_actual'] = v3_df['clarification_need'].apply(lambda x: 0 if x in [1, 2] else 1)

# Compute accuracy
accuracy = accuracy_score(v3_df['binary_actual'], v3_df['binary_predicted'])
print(f"Binary Classification Accuracy: {accuracy:.4f}")

# Compute confusion matrix
conf_matrix = confusion_matrix(v3_df['binary_actual'], v3_df['binary_predicted'])

# Convert confusion matrix to DataFrame for readability
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print("Binary Confusion Matrix:")
print(conf_matrix_df)
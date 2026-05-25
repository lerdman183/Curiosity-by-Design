import pandas as pd
import matplotlib.pyplot as plt

# Load training dataset
train_file = 'train.tsv'
train_df = pd.read_csv(train_file, sep='\t', dtype=str)

# Ensure 'clarification_need' column exists
if 'clarification_need' not in train_df.columns:
    raise ValueError("Column 'clarification_need' must be present in the training file.")

# Convert clarification_need to numeric
train_df['clarification_need'] = pd.to_numeric(train_df['clarification_need'], errors='coerce')

# Count occurrences of each label
distribution = train_df['clarification_need'].value_counts().sort_index()

# Plot bar graph
plt.figure(figsize=(8, 5))
plt.bar(distribution.index.astype(str), distribution.values, color='skyblue')
plt.xlabel('Clarification Need Label')
plt.ylabel('Count')
plt.title('Distribution of Clarification Need Labels in Training Set')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
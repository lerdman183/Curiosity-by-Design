import pandas as pd
import numpy as np
from collections import Counter

def shannon_entropy(text):
    """Calculate Shannon entropy of a given text."""
    if not text:
        return 0
    
    freq = Counter(text)  # Count character occurrences
    total_chars = len(text)
    entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in freq.values())
    return entropy

# Load training and testing datasets
train_file = 'train.tsv'
test_file = 'test.tsv'

try:
    train_df = pd.read_csv(train_file, sep='\t', dtype=str)
    test_df = pd.read_csv(test_file, sep='\t', dtype=str)
    
    # Ensure 'initial_request' column exists
    if 'initial_request' not in train_df.columns or 'initial request' not in test_df.columns:
        raise ValueError("Column 'initial_request' must be present in both files.")
    
    # Compute entropy for each request
    train_df['entropy'] = train_df['initial_request'].apply(shannon_entropy)
    test_df['entropy'] = test_df['initial request'].apply(shannon_entropy)
    
    # Compute average entropy
    avg_train_entropy = train_df['entropy'].mean()
    avg_test_entropy = test_df['entropy'].mean()
    
    print(f"Average Shannon Entropy in Training Set: {avg_train_entropy:.4f}")
    print(f"Average Shannon Entropy in Testing Set: {avg_test_entropy:.4f}")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
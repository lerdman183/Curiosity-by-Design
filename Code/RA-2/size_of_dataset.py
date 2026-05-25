import pandas as pd

# Load training and testing datasets
train_file = 'train.tsv'
test_file = 'test.tsv'

try:
    train_df = pd.read_csv(train_file, sep='\t', dtype=str)
    test_df = pd.read_csv(test_file, sep='\t', dtype=str)
    
    # Display dataset sizes
    print(f"Training set size: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Testing set size: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
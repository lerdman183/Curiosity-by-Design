import pandas as pd

# Load train.tsv and test.tsv
train_df = pd.read_csv('train.tsv', sep='\t', dtype=str)
test_df = pd.read_csv('test.tsv', sep='\t', dtype=str)

# Extract the relevant fields
train_initial_requests = set(train_df['initial_request'].dropna())
test_initial_requests = set(test_df['initial request'].dropna())

# Find overlap
overlap = train_initial_requests & test_initial_requests

# Print results
if overlap:
    print(f"Found {len(overlap)} overlapping entries:")
    for item in list(overlap)[:10]:  # Print first 10 overlaps as a sample
        print(item)
else:
    print("No overlap found.")
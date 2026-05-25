import pandas as pd

# Load test_with_labels.tsv
test_df = pd.read_csv('test_with_labels.tsv', sep='\t', dtype=str)

# Drop duplicates based on 'initial_request' while keeping the first occurrence
test_unique_df = test_df[['initial_request', 'clarification_need']].drop_duplicates(subset=['initial_request'])

# Load V3_predicted_clarification_need_from_scratch.tsv (if exists)
try:
    v3_df = pd.read_csv('V3_predicted_clarification_need_from_scratch.tsv', sep='\t', dtype=str)
except FileNotFoundError:
    v3_df = pd.DataFrame(columns=['initial_request', 'predicted_clarification_need'])

# Append the unique data
v3_df = pd.concat([v3_df, test_unique_df], ignore_index=True)

# Save back to file
v3_df.to_csv('V3_predicted_clarification_need_from_scratch.tsv', sep='\t', index=False)

print("Unique 'initial_request' entries added to V3_predicted_clarification_need_from_scratch.tsv.")
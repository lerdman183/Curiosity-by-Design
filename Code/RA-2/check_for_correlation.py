import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
file_path = 'V3_predicted_clarification_need_from_scratch.tsv'
df = pd.read_csv(file_path, sep='\t', dtype=str)

# Ensure required columns exist
if 'initial request' not in df.columns or 'predicted_clarification_need' not in df.columns:
    raise ValueError("Columns 'initial request' and 'predicted_clarification_need' must be present in the file.")

# Drop rows with missing values
df = df.dropna(subset=['initial request', 'predicted_clarification_need'])

# Compute input length
df['input_length'] = df['initial request'].apply(len)

# Convert predicted clarification need to numeric
df['predicted_clarification_need'] = df['predicted_clarification_need'].astype(int)

# Compute correlation
corr, p_value = pearsonr(df['input_length'], df['predicted_clarification_need'])
print(f"Pearson correlation: {corr:.4f}, P-value: {p_value:.4f}")

# Plot correlation
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['input_length'], y=df['predicted_clarification_need'], alpha=0.5)
plt.xlabel('Input Length (characters)')
plt.ylabel('Predicted Clarification Need')
plt.title('Correlation between Input Length and Predicted Clarification Need')
plt.show()

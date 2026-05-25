import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
test_file = 'test.tsv'

# Read the test data
test_data = pd.read_csv(test_file, sep='\t')

# Extract features
X_test = test_data['initial request']

# Text vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_test_tfidf = vectorizer.fit_transform(X_test)

# Load a pre-trained model (assume it's already trained and saved as 'model.pkl')
import joblib
model = joblib.load('model_basic_v1.pkl')

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Save predictions to a file
predictions = test_data.copy()
predictions['predicted_clarification_need'] = y_pred
predictions.to_csv('V2_predicted_clarification_need.tsv', sep='\t', index=False)

print("Predictions saved to V2_predicted_clarification_need.tsv")

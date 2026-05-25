# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# # from sklearn.model_selection import train_test_split

# # Load datasets
# train_file = 'train.tsv'
# test_file = 'test.tsv'
# test_with_labels_file = 'test_with_labels.tsv'

# train_data = pd.read_csv(train_file, sep='\t')
# test_data = pd.read_csv(test_file, sep='\t')
# test_with_labels = pd.read_csv(test_with_labels_file, sep='\t')

# # Extract features and labels
# X_train = train_data['initial_request']
# y_train = train_data['clarification_need']
# X_test = test_data['initial request']
# y_test = test_with_labels['clarification_need']

# # Text vectorization
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Train the model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train_tfidf, y_train)

# # Make predictions
# y_pred = model.predict(X_test_tfidf)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:\n", report)

# # Save predictions to a file
# predictions = test_data.copy()
# predictions['predicted_clarification_need'] = y_pred
# predictions.to_csv('predicted_clarification_need.tsv', sep='\t', index=False)

# print("Predictions saved to predicted_clarification_need.tsv")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
import joblib

# Load dataset
train_file = 'train.tsv'
test_file = 'test.tsv'

# Read the train and test data
train_data = pd.read_csv(train_file, sep='\t')
test_data = pd.read_csv(test_file, sep='\t')

# Extract features and labels for training
X_train = train_data['initial_request']
y_train = train_data['clarification_need']

# Extract features for testing
X_test = test_data['initial request']

# Text vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(model, 'model_basic_v3.pkl')

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Save predictions to a file
predictions = test_data.copy()
predictions['predicted_clarification_need'] = y_pred
predictions.to_csv('V3_predicted_clarification_need_from_scratch.tsv', sep='\t', index=False)

print("Predictions saved to V3_predicted_clarification_need_from_scratch.tsv")


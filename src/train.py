import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('data/tickets_cleaned.csv')

# Prepare data
X = df['ticket_description']
y = df['category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create ML Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100)),  # Convert text to numbers
    ('classifier', MultinomialNB())  # Classification algorithm
])

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)
print("Model trained!")

# Test model
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Category')
plt.xlabel('Predicted Category')
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png')
print("\nConfusion matrix saved: screenshots/confusion_matrix.png")

# Save model
with open('model/ticket_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to: model/ticket_classifier.pkl")
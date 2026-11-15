import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Remove missing values
data = data.dropna(subset=["text"])

# Input and output
X = data["text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=50000
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train AI model
model = PassiveAggressiveClassifier(
    max_iter=50,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100)
print(classification_report(y_test, predictions))

# Save trained model
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved successfully!")

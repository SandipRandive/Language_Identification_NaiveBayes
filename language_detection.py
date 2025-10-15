# =========================================================
# Project: Language Identification using Naive Bayes
# Dataset: Kaggle - Language Detection (17 Languages)
# =========================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
print("Loading dataset...")
data = pd.read_csv("C:/Users/Sandip/Downloads/Language Detection.csv (1)/Language Detection.csv")

print("\nDataset loaded successfully!")
print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())

# Step 3: Data Cleaning
print("\nChecking for missing values...")
print(data.isnull().sum())

# Drop duplicates if any
data.drop_duplicates(inplace=True)
print("After removing duplicates:", data.shape)

# Convert text to lowercase and remove special characters
data['Text'] = data['Text'].str.lower()
data['Text'] = data['Text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Step 4: Data Visualization and Statistical Summary
data['text_length'] = data['Text'].apply(len)

mean_len = data['text_length'].mean()
median_len = data['text_length'].median()
mode_len = data['text_length'].mode()[0]

print(f"\nMean text length: {mean_len:.2f}")
print(f"Median text length: {median_len}")
print(f"Mode text length: {mode_len}")

# Count of each language
plt.figure(figsize=(12,5))
sns.countplot(x='Language', data=data, palette='coolwarm', order=data['Language'].value_counts().index)
plt.title("Language Distribution")
plt.xticks(rotation=45)
plt.show()

# Box plot for text length by language
plt.figure(figsize=(12,6))
sns.boxplot(x='Language', y='text_length', data=data, palette='Set3')
plt.xticks(rotation=45)
plt.title("Box Plot of Text Length by Language")
plt.show()

# Step 5: Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])
y = data['Language']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nData split into training and testing sets successfully.")

# Step 7: Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Model Evaluation
print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Language")
plt.ylabel("Actual Language")
plt.show()

# Step 10: Test with Custom Sentences
sample_texts = [
    "Bonjour, comment ça va?",
    "Hello, nice to meet you!",
    "Hola amigo, cómo estás?",
    "Guten Morgen, wie geht’s?",
    "Ciao, come stai?",
    "Bom dia meu amigo!"
]

sample_features = vectorizer.transform(sample_texts)
predicted_languages = model.predict(sample_features)

print("\nCustom Sentence Predictions:")
for text, lang in zip(sample_texts, predicted_languages):
    print(f"'{text}'  →  {lang}")

print("\n✅ Language Identification using Naive Bayes completed successfully!")

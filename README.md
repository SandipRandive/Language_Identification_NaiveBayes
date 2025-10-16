# Language_Identification_NaiveBayes
A machine learning project for identifying text language using Naive Bayes.
# 🧠 Language Identification using Naive Bayes

This project focuses on building a **Machine Learning model** that can automatically **detect the language** of a given text input using the **Multinomial Naive Bayes algorithm**.  
It performs **data preprocessing, feature extraction, visualization, and evaluation** to achieve accurate and interpretable language predictions.

---

## 🌟 Project Overview

In a multilingual world, identifying the language of text is crucial for applications like translation, speech recognition, and content filtering.  
This project applies **Natural Language Processing (NLP)** and **Naive Bayes classification** to categorize text into different languages such as English, Hindi, French, Spanish, and more.

The model learns from a labeled dataset containing sample sentences in various languages, extracts textual features, and predicts the language of unseen data with high accuracy.

---

## 🎯 Objectives

- To develop an accurate **language classification model** using Naive Bayes  
- To perform **data cleaning, preprocessing, and feature engineering**  
- To visualize dataset characteristics using **matplotlib and seaborn**  
- To evaluate performance using **accuracy, confusion matrix, and other metrics**

---

## ⚙️ Features

✅ Detects the language of input text with high accuracy  
✅ Uses **Bag of Words (BoW)** or **TF-IDF** features  
✅ Performs **data cleaning, preprocessing, and visualization**  
✅ Displays **mean, median, and mode** for text statistics  
✅ Generates **confusion matrix** and **performance plots**  
✅ Fully modular, easy to extend to more languages  
✅ Clean and simple Python implementation  

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.x |
| Machine Learning | scikit-learn |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Text Processing | re, nltk (optional) |
| Model Storage | pickle / joblib |

---

## 🧩 Project Workflow

1. **Data Collection**  
   - CSV dataset containing text samples and their language labels  
   - Example columns: `Text`, `Language`

2. **Data Preprocessing**  
   - Text normalization (lowercasing, punctuation removal)  
   - Handling missing or duplicate entries  

3. **Feature Extraction**  
   - Convert text to numerical features using **CountVectorizer** or **TF-IDF Vectorizer**  

4. **Model Training**  
   - Use **Multinomial Naive Bayes** to train on the dataset  
   - Evaluate with accuracy score and confusion matrix  

5. **Model Evaluation & Visualization**  
   - Visualize data distributions and confusion matrix  
   - Display model accuracy and performance metrics  

6. **Prediction**  
   - Input a new sentence → Model predicts its language  

---

## 📂 Folder Structure

---

## 💻 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SandipRandive/Language_Identification_NaiveBayes.git
   cd Language_Identification_NaiveBayes



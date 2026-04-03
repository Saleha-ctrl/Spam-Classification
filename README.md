# 📧 Email Spam Classification System

## Overview
This project is a Machine Learning-based Email Spam Classification system that detects whether an email is **spam or not spam (ham)** using Natural Language Processing (NLP) techniques.

The model learns patterns from text data and automatically classifies incoming emails.

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorizer

---

## Problem Statement
Spam emails are a major issue in communication systems.  
The goal of this project is to build a model that can accurately classify emails as:
- Spam 📩
- Not Spam (Ham) 📬

---

## Workflow
1. Load email dataset
2. Clean and preprocess text data
3. Remove stopwords and special characters
4. Convert text into numerical features using TF-IDF
5. Train ML models (Naive Bayes / Logistic Regression)
6. Evaluate performance
7. Predict new email messages

---

## Approach
- Text preprocessing (lowercasing, tokenization, cleaning)
- Feature extraction using TF-IDF
- Model training using classification algorithms
- Performance evaluation using accuracy, precision, recall, F1-score

---

## Results
- High accuracy achieved on test dataset
- Strong performance in detecting spam messages
- Balanced precision and recall for classification

---

## Sample Prediction

**Input Email:**
> "Congratulations! You have won a free iPhone. Click the link to claim now."

**Output:**
>  Spam Email

---

## How to Run

```bash
pip install -r requirements.txt
python data.py

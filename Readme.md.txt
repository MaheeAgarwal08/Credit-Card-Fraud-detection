# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using Logistic Regression, SMOTE, and StandardScaler.

---

## 📌 Problem Statement

Credit card fraud is rare but highly impactful. The dataset is heavily imbalanced, so we apply techniques like SMOTE and Logistic Regression to detect fraud with high recall.

---

## 🧠 Approach

- Dataset: [Kaggle Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Preprocessing:
  - Train-test split with stratified sampling
  - Feature scaling with `StandardScaler`
  - SMOTE for oversampling the minority class (fraud)
- Model: `LogisticRegression` with solver `'liblinear'`
- Evaluation:
  - Confusion Matrix
  - ROC-AUC Curve
  - Classification Report

---

## 📊 Metrics

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 98.99%  |
| Fraud Recall  | 90%     |
| AUC-ROC Score | 0.98    |

---

## 🧪 Sample Predictions


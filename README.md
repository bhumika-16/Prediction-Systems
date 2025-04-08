# 🤖 Prediction Systems

This repository contains two machine learning projects that demonstrate practical implementation of classification and regression models using real-world datasets.

---

## 1. 📧 Email Spam Filtering

**Goal:** Build a robust machine learning model to classify emails as spam or not spam based on word-level features.

### 🔍 Problem Statement
Given a dataset of 5,172 emails with 3,000 word columns and 1 label column, the objective is to classify messages as spam (1) or not spam (0).

### 📌 Highlights
- **Features:** Frequency of words in each email
- **Labels:** Spam = 1, Not Spam = 0
- **Tools:** Python, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn

### 🧠 Algorithms & Performance
| Algorithm                | Accuracy  |
|--------------------------|-----------|
| Naïve Bayes              | 95.87%    |
| Support Vector Machine   | 96.33%    |
| **Random Forest**        | **96.71%** ✅ |

### 📈 Outcome
- Random Forest yielded the highest performance.
- Visualizations such as box plots, histograms, and confusion matrices used for evaluation.
- Data was preprocessed using normalization and encoding techniques.

---

## 2. 📱 App Rating Prediction (Apple Store)

**Goal:** Predict user ratings of Apple Store apps using metadata features via regression models.

### 🔍 Problem Statement
Predict the average app rating using attributes like size, price, genre, supported devices, screenshots, and version count.

### 📌 Highlights
- **Dataset:** 7,197 apps with 16 features
- **Target Variable:** Average User Rating
- **Tools:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

### 🧠 Algorithms & Performance
| Algorithm                | Accuracy  |
|--------------------------|-----------|
| Linear Regression        | 11.37%    |
| Decision Tree Regressor  | 83.74%    |
| **Random Forest**        | **97.64%** ✅ |

### 📈 Outcome
- Random Forest provided the most accurate results.
- Applied label encoding, standardization, and correlation-based feature selection.
- Regression metrics used: MAE, MSE, R² score.

---

## 🛠️ Common Tech Stack

- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn  
- **Algorithms:** Naïve Bayes, SVM, Decision Trees, Random Forest  
- **Techniques:** Data preprocessing, classification, regression, model evaluation  
- **Tools:** Anaconda, Jupyter Notebook, scikit-learn pipelines

---

## 📌 Future Improvements

- Expand datasets to improve generalization
- Deploy models via REST APIs using Flask or FastAPI
- Perform hyperparameter tuning (GridSearchCV)
- Integrate deep learning models for comparison

---

## 👩‍💻 Author

**Bhumika Jindal**  
M.Tech CSE (AI/ML), IIIT Bangalore  
GitHub: [bhumika-16](https://github.com/bhumika-16)

---

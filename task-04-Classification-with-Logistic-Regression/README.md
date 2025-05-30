# Task 4: Classification with Logistic Regression

## 📌 Objective
Build a binary classifier using **Logistic Regression** to classify instances from a given dataset.

## 🧠 Tools & Libraries
- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## 📂 Dataset
- **Name**: Breast Cancer Wisconsin Dataset
- **Target Column**: `diagnosis` (B = Benign, M = Malignant)
- **Source**: Provided in `data.csv`

## 🔍 Problem Statement
We aim to predict whether a tumor is **malignant** or **benign** using 30 numerical features. This is a binary classification task where Logistic Regression is applied.

---

## 🛠️ Steps Performed

### 1. Data Preprocessing
- Removed irrelevant columns: `id`, `Unnamed: 32`
- Converted target values: `M` → 1, `B` → 0
- Split into training and testing sets
- Standardized features using `StandardScaler`

### 2. Model Training
- Trained a **Logistic Regression** model from `sklearn`
- Used default solver with increased `max_iter=1000` for convergence

### 3. Model Evaluation
- **Confusion Matrix**
- **Classification Report**: Precision, Recall, F1-Score
- **ROC Curve & AUC Score**

### 4. Threshold Tuning
- Visualized the **Sigmoid Function**
- Understood how threshold affects classification

---

## 📊 Visuals
- `Confusion_Matrix.png`
- `ROC_Curve.png`
- `Sigmoid_Curve.png`

---

## ✅ What I Learned
- Logistic Regression fundamentals
- Evaluation metrics for binary classification
- ROC Curve and AUC interpretation
- Sigmoid function and threshold tuning

---


---

### 🚀 Developed by:
**Shankar Irla**  
_Elevate Labs AI/ML Internship_


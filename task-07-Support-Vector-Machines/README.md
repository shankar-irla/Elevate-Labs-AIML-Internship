# Task 7: Support Vector Machines (SVM)

## 📌 Objective
Use **Support Vector Machines (SVMs)** for **linear and non-linear binary classification** using Scikit-learn. Apply hyperparameter tuning, visualization, and performance evaluation techniques.

---

## 🛠️ Tools & Libraries Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn (optional, for visualizations)

---

## 📂 Dataset Used
- **Breast Cancer Dataset**
- Format: CSV
- Target column indicates diagnosis (malignant/benign)

---

## 🔁 Step-by-Step Procedure

### 🔹 Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
```

---

### 🔹 Step 2: Load and Explore Dataset

```python
df = pd.read_csv("breast-cancer.csv")
print(df.head())
print(df.info())
```

---

### 🔹 Step 3: Preprocess the Data

```python
df.dropna(inplace=True)

label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['diagnosis'])

X = df.drop(['diagnosis', 'target'], axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 🔹 Step 4: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---

### 🔹 Step 5: Train SVM Models

```python
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
```

---

### 🔹 Step 6: Evaluate Model Performance

```python
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

print("Linear Kernel Report:\n", classification_report(y_test, y_pred_linear))
print("RBF Kernel Report:\n", classification_report(y_test, y_pred_rbf))
```

---

### 🔹 Step 7: Hyperparameter Tuning

```python
params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), params, cv=5, verbose=2)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
```

---

### 🔹 Step 8: Visualize Decision Boundary (PCA)

```python
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X_scaled)
X_train_2D, _, y_train_2D, _ = train_test_split(X_2D, y, test_size=0.2, random_state=42)

svm_vis = SVC(kernel='rbf')
svm_vis.fit(X_train_2D, y_train_2D)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Decision Boundary - RBF Kernel")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

plot_decision_boundary(svm_vis, X_train_2D, y_train_2D)
```

---

## ✅ Results

- **Achieved Accuracy:** _(Add your model's accuracy here)_
- **Best Parameters:** _(From GridSearchCV output)_
- **Linear vs RBF Kernel Performance:** _(Summarize classification report comparisons)_

---

## 📌 Conclusion

- SVM with **RBF kernel** performs better for non-linearly separable data.
- Hyperparameter tuning is crucial for best results.
- PCA helps in visualizing classification performance in 2D space.

---

### 🚀 Developed by:
**Shankar Irla**  
_Elevate Labs AI/ML Internship_
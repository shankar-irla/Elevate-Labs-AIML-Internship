# Task 8: Clustering with K-Means

## 📌 Objective
Perform **unsupervised learning** using **K-Means clustering** to discover customer segments.

---

## 🛠️ Tools & Libraries Used

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NumPy

---

## 📂 Dataset Used
- **Mall_Customers.csv**
- Features: CustomerID, Gender, Age, Annual Income, Spending Score
- We will use relevant numeric features (e.g., `Annual Income`, `Spending Score`) for clustering.

---

## 🔁 Step-by-Step Procedure

### 🔹 Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

---

### 🔹 Step 2: Load and Explore Dataset

```python
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.info())
```

---

### 🔹 Step 3: Preprocess the Data

```python
# Use only numerical columns for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Optional: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 🔹 Step 4: Find Optimal K (Elbow Method)

```python
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal K')
plt.show()
```

---

### 🔹 Step 5: Apply KMeans Clustering

```python
optimal_k = 5  # (Choose based on elbow method)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original data
df['Cluster'] = clusters
```

---

### 🔹 Step 6: Visualize Clusters

```python
# 2D Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2')
plt.title('Customer Segments Based on K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()
```

---

### 🔹 Step 7: (Optional) PCA for Visualizing in 2D

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Set1')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA View of Clusters')
plt.show()
```

---

### 🔹 Step 8: Evaluate Clustering

```python
sil_score = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score: {sil_score:.3f}')
```

---

## ✅ Results

- **Optimal K:** _(e.g., 5 from elbow method)_
- **Silhouette Score:** _(value from step above)_
- **Interpretation:** Clusters represent customer segments like high income–low spending, low income–high spending, etc.

---

## 📌 Conclusion

- K-Means effectively groups customers into distinct segments.
- Elbow Method and Silhouette Score help validate cluster quality.
- Visualizations (PCA or 2D scatter plots) make segmentation insights easier to interpret.

---

## 📚 learning

- Clustering, Unsupervised Learning, Cluster Evaluation

---

### 🚀 Developed by:
**Shankar Irla**  
_Elevate Labs AI/ML Internship_

# 📊 Task 3: Linear Regression – Housing Price Prediction

## 🎯 Objective:
Implement and understand **simple & multiple linear regression** models to predict house prices.

---

## 🛠️ Tools Used:
- **Scikit-learn** – for model training and evaluation  
- **Pandas** – for data loading and preprocessing  
- **Matplotlib** & **Seaborn** – for data visualization  

---

## 🗂️ Dataset:
- **Name**: Housing.csv  
- **Path**: `./Housing.csv`  
- **Description**: Contains housing data with features like area, bedrooms, bathrooms, etc., to predict house prices.

---

## 🧾 Hints / Mini Guide:

1. **Import and Preprocess the Dataset**  
   - Load data using `pandas.read_csv()`
   - Handle categorical features using `get_dummies()`
   - Check for null values and outliers

2. **Split Data into Train-Test Sets**  
   - Use `train_test_split()` from `sklearn.model_selection`

3. **Fit a Linear Regression Model**  
   - Use `sklearn.linear_model.LinearRegression`

4. **Evaluate the Model**  
   - Use **MAE**, **MSE**, and **R² Score**
   - Visualize predictions using scatter plots

5. **Plot & Interpret Coefficients**  
   - Use bar plots to visualize feature importance

---

## 💻 Example Code Snippets

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Housing.csv')
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot coefficients
coeffs = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(10, 6))
sns.barplot(x=coeffs.values, y=coeffs.index)
plt.title("Feature Coefficients")
plt.show()

#Learning Outcome:
Data preprocessing techniques

Linear Regression modeling

Model evaluation metrics

Feature importance interpretation
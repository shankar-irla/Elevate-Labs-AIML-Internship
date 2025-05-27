## Task 2: Exploratory Data Analysis (EDA)

🔍 **Objective:**  
Understand the structure and patterns in a dataset using statistics and visualizations.

---

🧰 **Tools Used:**  
- **Pandas** – for data loading and manipulation  
- **Matplotlib**, **Seaborn** – for static visualizations  
- **Plotly** – for interactive visualizations  

---

📝 **Hints/Mini Guide:**

1. **Import libraries and load dataset**
2. **Generate summary statistics** (mean, median, std, nulls)
3. **Create histograms, boxplots, and count plots**
4. **Use correlation matrix or pairplots to explore relationships**
5. **Make feature-level insights from visuals**

---

📂 **Dataset**  
- **Used:** Titanic Dataset  
- **Path:** `../data/Titanic-Dataset.csv`  

---

💡 **Key Code Snippets**

```python
# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

```python
# 2. Load Dataset
df = pd.read_csv('../data/Titanic-Dataset.csv')
df.head()
```

```python
# 3. Summary Statistics
df.describe(include='all')  # Summary for all columns
df.info()
```

---
---

### 6: Feature-Level Inference (From Visuals)

🧠 **Observations:**

- 👩 **Females had a higher survival rate**  
  Visuals like count plots and survival rate by gender show that females were more likely to survive than males.

- 🛳️ **1st Class passengers were more likely to survive**  
  Boxplots and class-wise survival rate plots indicate that passengers in 1st class had a significantly higher chance of survival.

- 👶 **Most survivors were between 20–40 years old**  
  Histograms and age distribution plots reveal that the majority of survivors fell in the 20–40 age range.


🔎 **What You'll Learn:**

- Data Cleaning Essentials  
- Data Visualization Techniques  
- Basic Statistical Analysis  
- Identifying Trends, Outliers, and Correlations

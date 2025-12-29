import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Exercise 1 — Dataset Overview
    # head()
    # info()
    # describe()
    
data = {
    "age": [22, 25, 30, 35, 40, 45, 50],
    "salary": [30000, 35000, 50000, 60000, 65000, 70000, 120000],
    "experience": [1, 2, 5, 7, 10, 12, 20],
    "city": ["Colombo", "Kandy", "Colombo", "Galle", "Kandy", "Colombo", "Colombo"],
    "target": [0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

print("\nHead: ")
print(df.head())

print("\nInfo(): ")
print(df.info())

print("\nDescribe(): ")
print(df.describe())

# Exercise 2 — Numeric Distributions
    # Histogram
    # Boxplot
    
df["salary"].hist(bins=10)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

df.boxplot(column="salary")
plt.title("Salary Boxplot")
plt.show()

# Exercise 3 — Relationships
    # Scatter plot
    # Correlation matrix
    
df.plot.scatter(x="experience", y="salary")
plt.title("Experience vs Salary")
plt.show()

corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix: ")
print(corr)

#heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Exercise 4 — Categorical Analysis
    # value_counts()
    # Bar plot
    
print("\nCity Counts: ")
print(df["city"].value_counts())

# bar plot
df["city"].value_counts().plot(kind="bar")
plt.title("City Distribution")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()

df["target"].value_counts().plot(kind="bar")
plt.title("Target Distribution")
plt.show()
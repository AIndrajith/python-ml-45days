import numpy as np
import pandas as pd

data = {
    "age": [22, 28, 35, 45, 52],
    "salary": [30000, 45000, 60000, 80000, 150000],
    "experience": [1, 3, 7, 15, 25],
    "city": ["Colombo", "Kandy", "Colombo", "Galle", "Colombo"]
}

df = pd.DataFrame(data)
print(df)

# Exercise 1 — Create New Feature
    # Combine existing columns
    
df["salary_per_experience"] = df["salary"] / df["experience"]

print("\nSalary per experience: ")
print(df[["salary", "experience", "salary_per_experience"]])

# Exercise 2 — Binning (Discertization)
    # Create age groups
    
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 40, 60],
    labels=["Young", "Adult", "Senior"]
)

print("\nAge Groups: ")
print(df[["age", "age_group"]])

# Exercise 3 — Encoding
    # Encode categorical feature
    
df_encoded = pd.get_dummies(df, columns=["city"])

print("\nAfter one-hot encoding: ")
print(df_encoded)

# Exercise 4 — Transformation
    # Apply log transform
    
df_encoded["salary_log"] = np.log1p(df_encoded["salary"])

print("\nSalary before and after log transform: ")
print(df_encoded[["salary", "salary_log"]])
import pandas as pd
import numpy as np

# Exercise 1 — Create DataFrame
    # From dictionary
    # Print head, shape
    
data = {
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [25, 30, 35, 28],
    "score": [85, 90, 88, 92]
}

df = pd.DataFrame(data)
print(df.head())

print("\nShape:")
print(df.shape)

# Exercise 2 — Filtering
    # Filter rows based on condition
    
# Filter people older than 28
filtered_df = df[df["age"] > 28]

print("\nFiltered (age > 28): ")
print(filtered_df)

# Exercise 3 — Missing Values
    # Add NaN values
    # Fill them
    
df.loc[1, "age"] = np.nan
df.loc[3, "score"] = np.nan

print("\nWith missing values: ")
print(df)

print("\nMissing values count: ")
print(df.isnull().sum())

df["age"].fillna(df["age"].mean(), inplace=True)
df["score"].fillna(df["score"].mean(), inplace=True)

print("\nAfter filling missing values: ")
print(df)

# Exercise 4 — ML Split
    # Create X and y from DataFrame

# create target column
df["passed"] = df["score"] >= 85

# features and target
X = df[["age", "score"]]
y = df["passed"]

print("\nFeatures (X): ")
print(X)

print("\nTarget(y): ")
print(y)
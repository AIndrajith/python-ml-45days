import numpy as np
import pandas as pd

# Exercise 1 — Missing Values   #####################################
    # Detect missing values
    # Fill numeric with mean
    # Fill categorical with "Unknown"
    
data = {
    "age": [25, 30, np.nan, 40, 30],
    "salary": [50000, 60000, 55000, 120000, 60000],
    "city": ["Colombo", "Kandy", "Colombo", None, "Kandy"],
    "target": [1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
print("Original Dataframe: ")
print(df)

print("\nMissing values count: ")
print(df.isnull().sum())   

# Fill numeric with mean
df["age"] = df["age"].fillna(df["age"].mean())

# Fill categorical with "Unknown"
df["city"] = df["city"].fillna("Unknown")

# both can be written as like this
df = df.fillna({
    "age" : df["age"].mean(),
    "city" : "Unknown"
})

# Exercise 2 — Duplicates ###########################################
    # Find and remove duplicates
    
print("\nDuplicate rows: ")
print(df.duplicated())

df = df.drop_duplicates()

# Exercise 3 — Encoding     #########################################
    # Convert categorical column using one-hot encoding
    
df_encoded = pd.get_dummies(df, columns=["city"])

print("\nAfter One-Hot Encoding: ")
print(df_encoded)

# Exercise 4 — Scaling      #########################################
    # Apply min-max or standard scaling
    
df_encoded["salary_scaled"] = (
    df_encoded["salary"] - df_encoded["salary"].mean()
) / df_encoded["salary"].std()


# Final Ml split

X = df_encoded.drop("target", axis=1)
y = df_encoded["target"]

print("\nFeatures (X): ")
print(X)

print("\nTarget (y): ")
print(y)
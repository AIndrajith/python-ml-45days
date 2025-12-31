import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = {
    "age":[22, 25, 30, 35, 40, 45, 50, 55],
    "salary":[30000, 35000, 50000, 60000, 65000, 70000, 90000, 120000],
    "experience":[1, 2, 5, 7, 10, 12, 18, 25],
    "target":[0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "salary", "experience"]]
y = df["target"]

# Exercise 1 — Basic Split #################################
    # Split X and y
    # Print shapes
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 80 % train, 20% test
    random_state=42  # same split every run
)

print("Train shape: ", X_train.shape, y_train.shape)
print("Test shape: ", X_test.shape, y_test.shape)

# Exercise 2 — Scaling Without Leakage #####################
    # Scale train only
    # Transform test
    
scaler = StandardScaler()

# fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# apply same transform to test data
X_test_scaled = scaler.fit_transform(X_test)

# Exercise 3 — Stratified Split ############################
    # Use classification target
    
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, 
    test_size=.2,
    stratify=y,    # keeps class propotions same, 
    random_state=42
)

print("Target distribution (full): ")
print(y.value_counts())

print("\nTarget distribution (train): ")
print(y_train_s.value_counts())

print("\nTarget distribution (test):")
print(y_test_s.value_counts())

# Exercise 4 — Pipeline (Advanced but Important) ###########
    # Combine scaler + model
    
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print("Test accuracy: ",accuracy)
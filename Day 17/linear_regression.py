import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "experience": [1, 2, 3, 5, 7, 10, 12, 15],
    "age": [22, 25, 28, 30, 35, 40, 45, 50],
    "salary": [30000, 35000, 40000, 50000, 60000, 75000, 85000, 100000]
}

df = pd.DataFrame(data)
print(df)

# Exercise 1 — Simple Linear Regression
    # One feature
    # Plot line

X = df[["experience"]]   # 2D array
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# Exercise 2 — Multiple Linear Regression
    # Multiple features

# Exercise 3 — Evaluate Model
    # MAE, MSE, RMSE, R²

# Exercise 4 — Interpret Coefficients
    # Explain feature importance
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

data = {
    "age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "salary": [30000, 35000, 50000, 60000, 65000, 70000, 90000, 120000, 130000, 150000],
    "experience": [1, 2, 5, 7, 10, 12, 18, 25, 30, 35],
    "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "salary", "experience"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Exercise 1 — Decision Tree Classification
    # Train tree
    # Predict classes

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("Accuracy (unrestricted tree):", accuracy_score(y_test, y_pred))

# Exercise 2 — Control Overfitting
    # Limit max_depth
    # Compare performance
    
tree_limited = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)

tree_limited.fit(X_train, y_train)
y_pred_limited = tree_limited.predict(X_test)

print("Accuracy (limited tree):", accuracy_score(y_test, y_pred_limited))

# Exercise 3 — Feature Importance
    # Print feature importance

importances = pd.Series(
    tree_limited.feature_importances_,
    index=X.columns
)

print("\nFeature Importances:")
print(importances)

# Exercise 4 — Decision Tree Regression
    # Predict salary

X = df[["age", "experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

tree_reg = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Regression MSE:", mse)
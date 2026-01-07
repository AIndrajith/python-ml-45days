import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

# Exercise 1 — Random Forest Classification
    # Train forest
    # Evaluate accuracy
    
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Exercise 2 — Compare with Decision Tree
    # Show stability improvement

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Exercise 3 — Feature Importance
    # Print ranked features

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)

# Exercise 4 — Random Forest Regression
    # Predict salary
    
X = df[["age", "experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Random Forest Regression MSE:", mse)
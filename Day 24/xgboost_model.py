import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from xgboost import XGBClassifier, XGBRegressor

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

# Exercise 1 — XGBoost Classification===========================
    # Train classifier
    # Evaluate accuracy

xgb = XGBClassifier(
    n_estimators=100,       # ==> 100 boosted trees
    learning_rate=0.1,      # slow, stable learning
    max_depth=3,            # weak trees (prevents overfitting)
    subsample=0.8,
    colsample_bytree=0.8,   # subsample & colsample => randomness => better generalization
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

# Exercise 2 — Regularization Effect ===========================
    # Compare alpha/lambda

xgb_reg_strong = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    reg_alpha=1.0,     # L1 regularization  => removes useless splits
    reg_lambda=10.0,   # L2 regularization  => Penalizes large trees
    eval_metric="logloss",
    random_state=42
)

xgb_reg_strong.fit(X_train, y_train)

y_pred = xgb_reg_strong.predict(X_test)

print("Accuracy with strong regularization:", accuracy_score(y_test, y_pred))

# Exercise 3 — Feature Importance    ===========================
    # Print & interpret

importances = pd.Series(            # Importance = how much a feature reduces loss
    xgb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)

# Exercise 4 — XGBoost Regression    ===========================
    # Predict salary
    
X = df[["age", "experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# Train XGBoost regressor
xgb_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("XGBoost Regression MSE:", mse)
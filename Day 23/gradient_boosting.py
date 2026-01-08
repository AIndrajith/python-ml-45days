import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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

# Exercise 1 — Gradient Boosting Classification
    # Train classifier
    # Evaluate accuracy
    
gb = GradientBoostingClassifier(
    n_estimators=100,           # 100 small trees
    learning_rate=0.1,          # each tree makes small corrections
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))

# Exercise 2 — Effect of Learning Rate
    # Compare 0.1 vs 0.01
    
for lr in [0.1, 0.01]:
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Learning Rate={lr} → Accuracy={acc:.2f}")

# Exercise 3 — Feature Importance
    # Print important features
    
importances = pd.Series(
    gb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)

# Exercise 4 — Gradient Boosting Regression
    # Predict salary
    
X = df[["age", "experience"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_reg.fit(X_train, y_train)

y_pred = gb_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Gradient Boosting Regression MSE:", mse)
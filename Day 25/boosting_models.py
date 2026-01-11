import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import lightgbm as lgb
from catboost import CatBoostClassifier

data = {
    "age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "salary": [30000, 35000, 50000, 60000, 65000, 70000, 90000, 120000, 130000, 150000],
    "experience": [1, 2, 5, 7, 10, 12, 18, 25, 30, 35],
    "department": ["IT", "IT", "HR", "HR", "IT", "Sales", "Sales", "Sales", "IT", "HR"],
    "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop("target", axis=1)
y = df["target"]

# Exercises:

# 1️⃣ LightGBM classification ===========================
X_lgb = pd.get_dummies(X, columns=["department"])

X_train, X_test, y_train, y_test = train_test_split(
    X_lgb, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
# train lighgbm
start = time.time()

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)

end = time.time()

print("LightGBM Accuracy:", accuracy_score(y_test, y_pred))
print("LightGBM Training Time:", round(end - start, 4), "seconds")

# 2️⃣ CatBoost classification with categorical data ===========================

cat_features = ["department"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

start = time.time()

cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    verbose=False,
    random_state=42
)
# train catboost
cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

y_pred = cat_model.predict(X_test)

end = time.time()

print("CatBoost Accuracy:", accuracy_score(y_test, y_pred))
print("CatBoost Training Time:", round(end - start, 4), "seconds")

# 3️⃣ Compare training speed ===========================

print("\nModel Comparison:")
print("LightGBM → very fast, needs encoding")
print("CatBoost → slightly slower, handles categories natively")

# 4️⃣ Feature importance comparison ===========================

lgb_importance = pd.Series(
    lgb_model.feature_importances_,
    index=X_lgb.columns
).sort_values(ascending=False)

print("\nLightGBM Feature Importance:")
print(lgb_importance)

cat_importance = pd.Series(
    cat_model.get_feature_importance(),
    index=X.columns
).sort_values(ascending=False)

print("\nCatBoost Feature Importance:")
print(cat_importance)
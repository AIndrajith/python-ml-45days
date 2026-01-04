import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

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


# Exercise 1 — Binary Classification
    # Train logistic regression
    # Predict probabilities

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)

print("Predicted probabilities:")
print(y_prob)

y_pred = pipeline.predict(X_test)

print("\nPredicted classes:")
print(y_pred)

# Exercise 2 — Decision Threshold
    # Change threshold
    # Observe precision/recall

threshold = 0.7

y_pred_custom = (y_prob[:, 1] >= threshold).astype(int)

print("\nPredictions with threshold 0.7:")
print(y_pred_custom)

# Exercise 3 — Evaluate Model
    # Confusion matrix
    # Precision, Recall, F1

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nPrecision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Exercise 4 — Interpret Coefficients
    # Explain feature impact
    
model = pipeline.named_steps["model"]

coefficients = pd.Series(
    model.coef_[0],
    index=X.columns
)

print("\nFeature Coefficients:")
print(coefficients)    
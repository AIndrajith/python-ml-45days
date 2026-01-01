# Accuracy answers “How often am I right?”
# Precision & Recall answer “How am I wrong?”

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score
)

data = {
    "age": [22, 25, 30, 35, 40, 45, 50, 55],
    "salary": [30000, 35000, 50000, 60000, 65000, 70000, 90000, 120000],
    "experience": [1, 2, 5, 7, 10, 12, 18, 25],
    "target": [0, 0, 0, 1, 1, 1, 1, 1]
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

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Exercise 1 — Confusion Matrix
    # Generate predictions
    # Print confusion matrix
    
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# Exercise 2 — Precision, Recall, F1
    # Calculate all metrics
    
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Exercise 3 — Classification Report
    # Print full report
    
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Exercise 4 — ROC & AUC (Optional but valuable)
    # Plot ROC curve
    # Calculate AUC
    
y_prob = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("AUC Score:", auc_score)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")  # random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
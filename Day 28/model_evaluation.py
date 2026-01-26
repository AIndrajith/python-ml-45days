import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    accuracy_score
)

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# 1️⃣ Train classifier
numeric_features = ["age", "salary", "experience"]
categorical_features = ["department"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

pipeline.fit(X_train, y_train)

# 2️⃣ Confusion matrix
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 3️⃣ Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 4️⃣ ROC AUC score
y_proba = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", auc)

# optional ROC Curve (visual)
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.show()

# 5️⃣ Cross-validation
cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("Cross-validation F1 scores:", cv_scores)
print("Mean CV F1 score:", cv_scores.mean())

# 6️⃣ Detect overfitting
train_pred = pipeline.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
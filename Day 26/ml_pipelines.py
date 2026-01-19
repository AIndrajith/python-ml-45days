import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Exercise 1 — Build Full Pipeline
    # Numeric + categorical preprocessing
    # Classification model
    
# define column types
numeric_features = ["age", "salary", "experience"]
categorical_features = ["department"]

# create preprocessors
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# combining using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Exercise 2 — Train & Evaluate
    # Accuracy score
    # Clean predictions

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression())
    ]
)

# train pipeline
pipeline.fit(X_train, y_train)

# evaluate pipeline
y_pred = pipeline.predict(X_test)
print("Pipeline Accuracy:", accuracy_score(y_test, y_pred))

# Exercise 3 — Swap Model Easily
    # Replace model without changing preprocessing

# Replace Logistic Regression with Random Forest:
pipeline_rf = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

pipeline_rf.fit(X_train, y_train)

y_pred_rf = pipeline_rf.predict(X_test)
print("Random Forest Pipeline Accuracy:", accuracy_score(y_test, y_pred_rf))

# Exercise 4 — Predict on New Data
    # Simulate real-world usage
    
new_data = pd.DataFrame({
    "age": [28],
    "salary": [48000],
    "experience": [4],
    "department": ["IT"]
})

prediction = pipeline_rf.predict(new_data)

print("Prediction for new candidate:", prediction)
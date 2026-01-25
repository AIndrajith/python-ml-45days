
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Exercise 1 — Train a Pipeline
    # Goal: Build a pipeline with preprocessing + model.
    # Tasks
    # Use numeric + categorical features
    # Use any classifier (Logistic Regression / Random Forest)
    # You already know how from Day 26.
    
# Define column types
numeric_features = ["age", "salary", "experience"]
categorical_features = ["department"]

# Create preprocessors
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate Pipeline
y_pred = pipeline.predict(X_test)
print("Pipeline accuracy:", accuracy_score(y_test, y_pred))

# Exercise 2 — Save the Trained Pipeline
    # Goal: Persist the entire pipeline to disk.
    # Tasks
    # Use joblib
    # Save as model.joblib
    # This step makes your work reusable.
    
joblib.dump(pipeline, "model.joblib")
print("Model saved successfully!")


# Exercise 3 — Load the Pipeline Back
    # Goal: Load model without retraining.
    # Tasks
    # Load saved file
    # Store in a new variable
    
loaded_pipeline = joblib.load("model.joblib")
print("Model loaded successfully!")


# Exercise 4 — Predict Using Loaded Model
    # Goal: Ensure loaded model works exactly like original.
    # Tasks
    # Create new sample input
    # Predict using loaded pipeline

new_data = pd.DataFrame({
    "age": [28],
    "salary": [48000],
    "experience": [4],
    "department": ["IT"]
})

prediction_loaded = loaded_pipeline.predict(new_data)
print("Prediction using loaded model:", prediction_loaded)

# Exercise 5 — Validate Consistency (IMPORTANT)
    # Goal: Confirm predictions match.
    # Tasks
    # Predict using original pipeline
    # Predict using loaded pipeline
    
prediction_original = pipeline.predict(new_data)

print("Original pipeline prediction:", prediction_original)
print("Loaded pipeline prediction:", prediction_loaded)

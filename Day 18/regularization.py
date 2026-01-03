import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "experience": [1, 2, 3, 5, 7, 10, 12, 15],
    "age": [22, 25, 28, 30, 35, 40, 45, 50],
    "projects": [1, 1, 2, 3, 4, 6, 7, 9],
    "salary": [30000, 35000, 40000, 50000, 60000, 75000, 85000, 100000]
}

df = pd.DataFrame(data)

X = df[["experience", "age", "projects"]]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression Coefficients:", lr.coef_)

# Exercise 1 — Ridge Regression
    # Train & evaluate
    # Observe coefficients
    
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

print("\nRidge R²:", r2_score(y_test, y_pred_ridge))
print("Ridge Coefficients:", ridge.coef_)

# Exercise 2 — Lasso Regression
    # Identify removed features

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

print("\nLasso R²:", r2_score(y_test, y_pred_lasso))
print("Lasso Coefficients:", lasso.coef_)

# Exercise 3 — ElasticNet
    # Compare results

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)

print("\nElasticNet R²:", r2_score(y_test, y_pred_elastic))
print("ElasticNet Coefficients:", elastic.coef_)

# Exercise 4 — Alpha Tuning
    # Try different alpha values

for alpha in [0.01, 0.1, 1, 10]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    r2 = ridge.score(X_test_scaled, y_test)
    print(f"Alpha={alpha} → R²={r2:.3f}")
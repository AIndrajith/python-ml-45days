import pandas as pd

# Exercise 1 — GroupBy
    # Group by category
    
data = {
    "department": ["IT", "IT", "HR", "HR", "Sales", "Sales"],
    "salary": [60000, 65000, 50000, 52000, 70000, 72000]
}

df = pd.DataFrame(data)
print(df)

grouped = df.groupby("department")["salary"].agg(["mean", "count"])

print("\nGroupBy result: ")
print(grouped)

# Compute mean and count
    # Exercise 2 — Merge
    # Merge two DataFrames on ID
    
employees = pd.DataFrame({
    "emp_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

salaries = pd.DataFrame({
    "emp_id": [1, 2, 3],
    "salary": [60000, 65000, 70000]
})

merged_df = pd.merge(employees, salaries, on="emp_id")

print("\nMerged DataFrame: ")
print(merged_df)

# Exercise 3 — Apply
    # Create new column using logic
    
scores_df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [85, 72, 90]
})

# creating grade column
scores_df["grade"] = scores_df["score"].apply(
    lambda x: "Pass" if x >= 80 else "Fail"
)

print("\nScores with grades: ")
print(scores_df)

# Exercise 4 — Pivot Table
    # Summarize numeric data
    
sales = pd.DataFrame({
    "region": ["East", "East", "West", "West"],
    "product": ["A", "B", "A", "B"],
    "revenue": [100, 150, 200, 250]
})

pivot = pd.pivot_table(
    sales,
    values="revenue",
    index="region",
    columns="product",
    aggfunc="sum"
)

print("\nPivot Table:")
print(pivot)
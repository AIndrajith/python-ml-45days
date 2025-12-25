import numpy as np

# Exercise 1 — Broadcasting
    # Add a scalar to a vector
    # Add vector to matrix
    
arr = np.array([1,2,3]) 
print(arr + 10)

# Exercise 2 — Boolean Masking
    # Filter values greater than threshold
    
arr = np.array([10, 20, 5, 30, 15]) 
mask = arr > 15
filtered = arr[arr > 15]
print(filtered)

# Exercise 3 — Axis Operations
    # Row-wise mean
    # Column-wise sum
    
data = np.array([[1,2,3], [4,5,6]])

print(data.mean(axis=1))
print(data.sum(axis=0))

# Exercise 4 — Labels
    # Use np.where to convert values → 0/1
    
results = np.where(arr > 15,1,0)
print(results)
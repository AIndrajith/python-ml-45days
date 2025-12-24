# Exercise 1 — Create Arrays
    # 1D array
    # 2D array
    # Print shapes
    
import numpy as np

OneD_array = np.array([1,2,3])
TwoD_array = np.array([[1,2,3],
                       [4,5,6]])

print(OneD_array.shape)
print(TwoD_array.shape)

# Exercise 2 — Math Operations
    # Add two arrays
    # Multiply array by scalar
    
OneD_array2 = np.array([3,3,6])
print(OneD_array + OneD_array2)
print(OneD_array * TwoD_array)
print(OneD_array * 2)

# Exercise 3 — Reshape
    # Convert 1D → 2D
    # Print before & after shapes
    
arr1 = np.array([1,2,3,4,5,6])

print(f"Before converting to 2D: {arr1}")

arr2 = arr1.reshape(3,2)

print(f"After converting to 2D: {arr2}")

# Exercise 4 — Stats
    # Compute sum, mean, max

arr = np.array([1,2,3,4])

arr.sum()
arr.mean()
arr.max()
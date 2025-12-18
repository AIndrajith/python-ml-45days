# Exercise no 1 : Safe division
# Write a function that:
    # Divides two numbers
    # Catches division by zero
    # Returns None if error
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

result = safe_divide(18, 0)
print(result)   # None
    
# Exercise no 2: File loader
# Load a file safely:
    # Catch FileNotFoundError
    # Raise meaningful error message
    
def load_data(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset not found at {path}") from e
    

# Exercise no 3 : ML Input Validation
# Create a custom exception:
    # class InvalidDataError(Exception):
    #     pass
# Raise it if:
    # X or y is empty or None
    
class InvalidDataError(Exception):
    pass

def train_model(X, y):
    # if X is None or y is None or X == [] or y == []:
    if not X or not y:
        raise InvalidDataError("Training data cannot be None of empty")
    
    print("Training model...")
    
train_model([],[])
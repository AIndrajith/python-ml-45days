# Exersice no 1: Logging decorator
# create a decorator that prints:
    # Function <name> started
    # Function <name> ended
    
from functools import wraps

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} started")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} ended")
        return result
    return wrapper

@log_execution
def say_hello():
    print("Hello!")
    
say_hello()

# Exersice no 2: Timing decorator
# time any function and print duration

import time
from functools import wraps

def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@time_execution
def slow_function():
    time.sleep(1)
    
slow_function()

# Exersice no 3: Validation Decorator (ML-flavored)
# create a decorator that checks: 
#   No argument is None
#   if any is None, raise ValueError

from functools import wraps

def validate_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # check positional arguments
        for arg in args:
            if arg is None:
                raise ValueError("None vallue deteted in positional arguments")
        
        # check keyword arguments
        for key, value in kwargs.items():
            if value is None:
                raise ValueError(f"None value detected for argument '{key}'")
            
        return func(*args, **kwargs)
    return wrapper

@validate_inputs
def train_model(X, y):
    print("Training model...")
    
# valid call
train_model([1,2,3], [0,1,1])

# Invalid call (will raise error)
train_model(None, [0,1,1])
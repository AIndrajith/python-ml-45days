# Exercise no 1: File Context
# Read a file using "with" and print context

with open("a.txt","r") as file:
    for line in file:
        print(line)
        
with open("b.txt", "w") as f:
    f.write("test text")
    
    
# Exercise no 2: Custom Context manager(class)
# create a context manager that prints;
    # Start process
    # End process
    
class MyContext():
    def __enter__(self):
        print("Start Process")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("End process")
        
with MyContext():
    print("middle process")
    
# Exercise no 3: Custom Context Manager (Decorator way)
# Same as above, but using @contextmanager

from contextlib import contextmanager

@contextmanager
def my_context():
    print("Start")
    yield
    print("End")
    
with my_context():
    print("I am the middle one")
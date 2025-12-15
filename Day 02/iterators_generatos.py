# Exersice no 1:
# create an iterator that returns even numbers up to N

# for x in range(1,11):
#     if x % 2 == 0:
#         print(x)

class EvenNumbers:
    def __init__(self, max_value):
        self.current = 0
        self.max_value = max_value
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.max_value:
            raise StopIteration
        self.current += 2
        return self.current
    
for x in EvenNumbers(10):
    print(x, end=" ")
        
print("\n")        
# Exersice no 2:
# Generator function
def fibonacci(n):
    current_num = 0
    next_num = 1
    for i in range (1, n+1):
        yield current_num
        # this is called multiple assignment (or tuple unpacking)
        current_num , next_num = next_num, current_num + next_num
        
for num in fibonacci(7):
    print(num, end=" ")
    
print("\n")

# Exersice no 3:
# create a generator that yeilds squares of numbers only if divisible by 3
square = (x * x for x in range(1,20) if x % 3 == 0)

for value in square:
    print(value, end=" ")
    
# (expression for item in iterable if condition)
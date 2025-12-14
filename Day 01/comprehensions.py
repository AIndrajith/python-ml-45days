# Exercise 1 

nums = [1,2,3,4,5,6]
# create a list of cubes of odd numbers

odd_cubes = [ x ** 3 for x in nums if x % 2 == 1]
print(odd_cubes)

# Exercise 2

words = ["ml", "python", "ai", "engineer"]
# create dict: word -> length, only if length > 2

filtered = {n: len(n) for n in words if len(n) > 2}
print(filtered)

# Exercise 3

nums2 = [1,2,3,4]
# use lambda + reduce to calculate factorial of 4

from functools import reduce

factorial = reduce(lambda x,y: x * y, nums2 )
print(factorial)


# Exercese 4 (ML-flavored)

labels = [0,1,1,0,1]
# conver to dict: index -> label

label_dict = {index: label for index, label in enumerate(labels)}
print(label_dict)
# enumerate() gives => (index, value)
for i, v in enumerate(labels):
    print(i, v)
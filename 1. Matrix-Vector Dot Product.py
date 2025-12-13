""" 
1. Matrix-Vector Dot Product [Easy]

Write a Python function that computes the dot product of a matrix and a vector. 
The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. 
A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. 
For example, an n x m matrix requires a vector of length m.

Example:
Input:
a = [[1, 2], [2, 4]], b = [1, 2]
Output:
[5, 10]

Reasoning:
Row 1: (1 * 1) + (2 * 2) = 1 + 4 = 5; Row 2: (1 * 2) + (2 * 4) = 2 + 8 = 10

#Insights: 
 - I used numpy to make the solution faster rather than building it from scratch 

 #SOLUTION: 

 """ 

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	import numpy as np
    # Convert inputs to numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Validate dimensions
    # a must be 2D, b must be 1D, and inner dimensions must match
    if a.ndim != 2 or b.ndim != 1 or a.shape[1] != b.shape[0]:
        return -1

    # Compute dot product and return as a Python list
    return (a @ b).tolist()
	pass

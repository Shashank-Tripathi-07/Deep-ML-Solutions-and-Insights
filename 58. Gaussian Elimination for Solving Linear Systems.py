"""
58. Gaussian Elimination for Solving Linear Systems [Medium]

Task: Implement the Gaussian Elimination Method
Your task is to implement the Gaussian Elimination method, which transforms a system of linear equations into an upper triangular matrix.
This method can then be used to solve for the variables using backward substitution.
Write a function gaussian_elimination(A, b) that performs Gaussian Elimination with partial pivoting to solve the system (Ax = b).
The function should return the solution vector (x).

Example:
Input:
A = np.array([[2,8,4], [2,5,1], [4,10,-1]], dtype=float)
b = np.array([2,5,1], dtype=float)
print(gaussian_elimination(A, b))

Output:
[11.0, -4.0, 3.0]

Reasoning:
The Gaussian Elimination method transforms the system of equations into an upper triangular matrix and then uses backward substitution to solve for the variables.


Analysis and Insights: 

We'll follow the same approach as we do mathematically, just using code to get the job done. 
I'd suggest learning gaussian elimination if you don't know it. 

Rather than random row operations that we usually do in maths classes, here we follow a standardized approach 
and that is to swap two equations, multiply an equation by a non-zero constant and then to add a multiple of one equation to another. 
"""

#SOLUTION: 

import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	n = b.shape[0]

    # Forward elimination
    for k in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x.tolist()

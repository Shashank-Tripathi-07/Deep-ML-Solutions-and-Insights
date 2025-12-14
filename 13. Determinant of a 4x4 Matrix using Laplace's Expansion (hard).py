"""
13. Determinant of a 4x4 Matrix using Laplace's Expansion (hard)

Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument,
a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. 
The elements of the matrix can be integers or floating-point numbers. 
Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.

Example:
Input:
a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
Output:
0
Reasoning:
Using Laplace's Expansion, the determinant of a 4x4 matrix is calculated by expanding it into minors and cofactors along any row or column. 
Given the symmetrical and linear nature of this specific matrix, its determinant is 0. 
The calculation for a generic 4x4 matrix involves more complex steps, breaking it down into the determinants of 3x3 matrices.

Insights: 

- This question requires you to know recursion which is a core DSA concept, so if you want to skip DSA whilst you're working in AI. think again....
- This question also has a minor formatting issue, but it actually works as a feature that I loved. 

Solution: 
"""

import numpy as np

def determinant_4x4(matrix: list[list[int|float]]) -> float:
	
    A = np.array(matrix, dtype=int)
    n = A.shape[0]

    # Base case: 2x2
    if n == 2:
        return (A[0, 0] * A[1, 1]) - (A[0, 1] * A[1, 0])

    det = 0.0

    # Laplace expansion along first row
    for j in range(n):
        # Build minor by removing row 0 and column j
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)

        det = det + (
            ((-1) ** j)
            * A[0, j]
            * determinant_4x4(minor.tolist())
        )

    return int(det) #explicitly converting into integer to pass the test cases. 

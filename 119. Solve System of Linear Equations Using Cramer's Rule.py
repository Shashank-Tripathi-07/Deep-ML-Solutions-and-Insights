"""

119. Solve System of Linear Equations Using Cramer's Rule [Medium]

Implement a function to solve a system of linear equations 
Ax=b using Cramer's Rule. The function should take a square coefficient matrix 
A and a constant vector b, and return the solution vector x.
If the system has no unique solution (i.e., the determinant of A is zero), return -1.

Example:
Input:
A = [[2, -1, 3], [4, 2, 1], [-6, 1, -2]], b = [5, 10, -3]
Output:
[0.1667 3.3333 2.6667]


Insights: 

- A Basic mathematics to code implementation question 
- A good medium question 

"""


#SOLUTIONL 

import numpy as np

def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Validation of dimensions for edge cases. 
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape[0] != A.shape[0]:
        return -1

    det_A = np.linalg.det(A)

    # Code for No unique solution case 
    if np.isclose(det_A, 0.0):
        return -1

    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A

    return np.round(x, 4)    

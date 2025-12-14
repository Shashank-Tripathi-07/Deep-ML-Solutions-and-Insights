"""
11. Solve Linear Equations using Jacobi Method [Medium] 

Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

Example:
Input:
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
Output:
[0.146, 0.2032, -0.5175]

Reasoning:
The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)),
where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.


Insights: 

- A good look should be given to the jacobian matrix and learning about it beforehand actually helps. It helped me also. 
- It's a good problem, just take care of the formatting at the end. the return np.round(x,4).tolist() helps a lot in that case. 

"""

import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    m = A.shape[0]

    x = np.zeros(m)

    #Diagonal and remainder
    D = np.diag(A)
    R = A - np.diagflat(D) 

    for _ in range(n):
        x = (b - R @ x)/D


    return np.round(x,4).tolist()

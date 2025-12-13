"""

7. Matrix Transformation [Medium] 

Write a Python function that transforms a given matrix A using the operation ([T]^-1)*AS where T and S are invertible matrices.
The function should first validate if the matrices T and S are invertible, and then perform the transformation.
In cases where there is no solution return -1


Example:
Input:
A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
Output:
[[0.5,1.5],[1.5,3.5]]


Insights: 

#The edge cases were cool in this question 
# Spending time on the maths more than the code actually helps in this question. 


SOLUTION: 

"""

import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    t = np.array(T, dtype=float)
    s = np.array(S, dtype=float)
    a = np.array(A, dtype=float)
   
    # Check square matrices
    if t.shape[0] != t.shape[1] or s.shape[0] != s.shape[1]:
        return -1
   
    # Check invertibility using rank
    if np.linalg.matrix_rank(T) != t.shape[0]:
        return -1
    if np.linalg.matrix_rank(S) != s.shape[0]:
        return -1

    try:
        T_inv = np.linalg.inv(T)
        result = T_inv @ a @ s
        return result.tolist()
    except np.linalg.LinAlgError:
        return -1



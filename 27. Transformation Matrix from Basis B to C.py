"""
27. Transformation Matrix from Basis B to C [Easy] 

Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.

Example:
Input:
B = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]]
        C = [[1, 2.3, 3], 
             [4.4, 25, 6], 
             [7.4, 8, 9]]
Output:
[[-0.6772, -0.0126, 0.2342],
                [-0.0184, 0.0505, -0.0275],
                [0.5732, -0.0345, -0.0569]]
Reasoning:
The transformation matrix P from basis B to C can be found using matrix operations involving the inverse of matrix C.


Insights: 

- A good question to get the hold of the maths and the code, I recommend using the deep-ml environment to get a quick glance of theory for this one. 

Solution: 

"""

import numpy as np 

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    P = np.linalg.inv(C) @ B
	return P

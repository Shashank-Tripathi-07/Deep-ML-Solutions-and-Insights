"""
8. Calculate the inverse of a 2x2 matrix [Medium] 

Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

Example:
Input:
matrix = [[4, 7], [2, 6]]
Output:
[[0.6, -0.7], [-0.2, 0.4]]
Reasoning:
The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.


Insights: 

# I feel the only medium toughness is for the fact that both input and output are lists and we're calculating using numpy that too in float. 
# Also, the result gets submitted in the first try when the results are in float. but, a rounding off is needed to get the same answer as output required. 
# It's a fun question 

SOLUTION: 

"""

import numpy as np 

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

    arr = np.array(matrix, dtype=float)

    try: 
        inv = np.linalg.inv(arr)
        return (np.round(inv,2)).tolist()
    except LinAlgError :
        return "None" 



        

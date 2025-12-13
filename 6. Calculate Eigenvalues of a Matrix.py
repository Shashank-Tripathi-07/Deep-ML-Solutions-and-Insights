"""
Calculate Eigenvalues of a Matrix 

A medium question on DeepML 

Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

Example:
Input:
matrix = [[2, 1], [1, 2]]
Output:
[3.0, 1.0]
Reasoning:
The eigenvalues of the matrix are calculated using the characteristic equation of the matrix.

Insights:: 

# The more you spend time with the problem the easier the solution becomes....the hardest thing is for a programmer is to think on the problem when he can just hop and built a shit solution anyways...
#                                                                                                                                                                                            ~ Rocky 

# we can do sorted = True to reverse the ascending list that sorted function returns.
# I first submitted the simple solution without doing float which lead to the solution getting returned as an array rather than a float value. 
# The sorted_eigvals needed to be explicitly converted into a float value outside the numpy float to get a clean answer. 

SOLUTION : 

"""

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np 

    arr = np.array(matrix)
    eigenvalues = np.linalg.eigvals(arr)
    sorted_eigvals = sorted(eigenvalues, reverse=True) 

    return [float(x) for x in sorted_eigvals]

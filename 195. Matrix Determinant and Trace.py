"""
195. Matrix Determinant & Trace [Easy]

Implement a function that computes both the determinant and trace of a square matrix. 
The determinant is a scalar value that can be computed from the elements of a square matrix and encodes certain properties of the matrix.
The trace is simply the sum of the elements on the main diagonal.
Return both values as a tuple.

Example:
Input:
matrix=[[2, 3], [1, 4]]
Output:
(5.0, 6.0)


Insights: 

- another great basic question. 
- Remember np.linal.det() and np.trace to get the work done

"""


#SOLUTION: 

import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	A = np.array(matrix, dtype=float)

	#validate square matrix for edge case test
	if A.ndim != 2 or A.shape[0] != A.shape[1]:
		return -1

	determinant = float(np.linalg.det(A))
	trace = float(np.trace(A))

	return(determinant,trace)

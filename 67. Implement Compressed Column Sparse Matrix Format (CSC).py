"""
67. Implement Compressed Column Sparse Matrix Format (CSC) [Easy]

Task: Create a Compressed Column Sparse Matrix Representation

Your task is to implement a function that converts a dense matrix into its Compressed Column Sparse (CSC) representation.
The CSC format stores only non-zero elements of the matrix and is efficient for matrices with a high number of zero elements.
Write a function compressed_col_sparse_matrix(dense_matrix) that takes in a two-dimensional list dense_matrix and returns a tuple of three lists:

values: List of non-zero elements, stored in column-major order.
row indices: List of row indices corresponding to each value in the values array.
column pointer: List that indicates the starting index of each column in the values array.

Example:
Input:
dense_matrix = [
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
    ] 

vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)
Output:
[1, 2, 3, 4] [1, 2, 0, 1] [0, 1, 2, 3, 4]

Reasoning:
The dense matrix is converted to CSC format with the values array containing non-zero elements, row indices array storing the corresponding row index, 
and column pointer array indicating the start of each column in the values array.


Insights: 
- It may look like question 65 but upon a closer look you'll reaize both are different. 
- I've taken the same numpy approach as before, if you find the implementation tough like not understanding concatenate or bincount then I'd say check the Numpy documentation 
"""

#SOLUTION: 

import numpy as np 

def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	A = np.asarray(dense_matrix)

    # 1. Find non-zero positions (row indices, column indices)
    row_idx, col_idx = np.nonzero(A)

    # 2. Sort by column index (CSC requires column-major order)
    order = np.argsort(col_idx)
    row_idx = row_idx[order]
    col_idx = col_idx[order]

    # 3. Extract values in column-major order
    values = A[row_idx, col_idx]

    # 4. Build column pointer
    col_counts = np.bincount(col_idx, minlength=A.shape[1])
    col_ptr = np.concatenate(([0], np.cumsum(col_counts)))

    return values.tolist(), row_idx.tolist(), col_ptr.tolist()

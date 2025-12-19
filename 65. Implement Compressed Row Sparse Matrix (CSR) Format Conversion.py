"""

65. Implement Compressed Row Sparse Matrix (CSR) Format Conversion [Easy] 

Task: Convert a Dense Matrix to Compressed Row Sparse (CSR) Format
Your task is to implement a function that converts a given dense matrix into the Compressed Row Sparse (CSR) format, an efficient storage representation for sparse matrices. The CSR format only stores non-zero elements and their positions, significantly reducing memory usage for matrices with a large number of zeros.

Write a function compressed_row_sparse_matrix(dense_matrix) that takes a 2D list dense_matrix as input and returns a tuple containing three lists:

Values array: List of all non-zero elements in row-major order.
Column indices array: Column index for each non-zero element in the values array.
Row pointer array: Cumulative number of non-zero elements per row, indicating the start of each row in the values array.
Example:
Input:
dense_matrix = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 4, 0],
    [1, 0, 0, 5]
    ]

vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)
Output:
Values array: [1, 2, 3, 4, 1, 5]
Column indices array: [0, 1, 0, 2, 0, 3]
Row pointer array: [0, 1, 2, 4, 6]
Reasoning:
The dense matrix is converted to CSR format with the values array containing non-zero elements, column indices array storing the corresponding column index, and row pointer array indicating the start of each row in the values array.

Explanation:
The values array holds the non-zero elements in the matrix, in row-major order.
The column indices array stores the corresponding column index of each non-zero element.
The row pointer array keeps track of where each row starts in the values array. For example, row 1 starts at index 0, row 2 starts at index 1, row 3 starts at index 2, and so on.

Applications:
The CSR format is widely used in high-performance computing applications such as:

-Finite element analysis (FEA)
-Solving large sparse linear systems (e.g., in numerical simulations)
-Machine learning algorithms (e.g., support vector machines with sparse input)
-Graph-based algorithms where adjacency matrices are often sparse
-The CSR format improves both memory efficiency and the speed of matrix operations by focusing only on non-zero elements.

Insights: 

This method is great, especially when we're working on large matrices and calculations and have less usable memory [RAM] 
I've used a simple method of initialization, loops and result counts. check the comment in the code and you'll know. 
"""

#SOLUTION: 

import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
    values = []             #initializing required things to zero. 
    col_idx = []
    row_ptr = [0]

    count = 0  # number of non-zero elements seen so far in the input.

    for row in dense_matrix:
        for j, val in enumerate(row):           
            if val != 0:
                values.append(val)
                col_idx.append(j)
                count += 1
        row_ptr.append(count)

    return values, col_idx, row_ptr


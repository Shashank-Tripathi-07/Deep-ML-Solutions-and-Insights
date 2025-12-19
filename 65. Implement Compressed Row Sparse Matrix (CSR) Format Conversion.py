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
Check the solution code for a smarter solution 
"""

#SOLUTION: 

A = np.asarray(dense_matrix)

    # 1. Find non-zero positions
    row_idx, col_idx = np.nonzero(A)

    # 2. Extract values
    values = A[row_idx, col_idx]

    # 3. Build row pointer
    row_counts = np.bincount(row_idx, minlength=A.shape[0])
    row_ptr = np.concatenate(([0], np.cumsum(row_counts)))

    return values.tolist(), col_idx.tolist(), row_ptr.tolist()

# I can solve it throught the traditional DSA O(n^2) approach but I've taked this approach
# with numpy as the calculations are in C and hence are faster and better. 
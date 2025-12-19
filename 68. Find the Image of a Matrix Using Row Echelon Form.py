"""
68. Find the Image of a Matrix Using Row Echelon Form [Medium]

Task: Compute the Column Space of a Matrix
In this task, you are required to implement a function matrix_image(A) that calculates the column space of a given matrix A. The column space, also known as the image or span, consists of all linear combinations of the columns of A. To find this, you'll use concepts from linear algebra, focusing on identifying independent columns that span the matrix's image. Your task: Implement the function matrix_image(A) to return the basis vectors that span the column space of A. These vectors should be extracted from the original matrix and correspond to the independent columns.

Example:

Input:
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ])
print(matrix_image(matrix))
Output:
# [[1, 2],
#  [4, 5],
#  [7, 8]]

Reasoning:
The column space of the matrix is spanned by the independent columns [1, 2], [4, 5], and [7, 8]. These columns form the basis vectors that represent the image of the matrix.

Insights: 

-It contains a hidden check in the test case that actually teaches you some good learning if you're using the environment. 
- requires a few first tries. 
- Yeah, it's medium for a reason. 
"""

#SOLUTION: 

import numpy as np

def matrix_image(A):
    m, n = A.shape

    basis_cols = []
    current_matrix = np.empty((m, 0))

    current_rank = 0  # we track rank manually

    for j in range(n):
        candidate = np.column_stack((current_matrix, A[:, j]))

        new_rank = np.linalg.matrix_rank(candidate)

        # Only keep column if rank increases
        if new_rank > current_rank:
            basis_cols.append(A[:, j])
            current_matrix = candidate
            current_rank = new_rank

    return np.column_stack(basis_cols)

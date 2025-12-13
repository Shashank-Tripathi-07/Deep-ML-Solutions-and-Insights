"""
2. Transpose of a Matrix [literally the easiest one :) ] 

Write a Python function that computes the transpose of a given matrix.

Example:

Input:
a = [[1,2,3],[4,5,6]]
Output:
[[1,4],[2,5],[3,6]]

Reasoning:
The transpose of a matrix is obtained by flipping rows and columns.

Insights: 

# You can solve it in 2 lines using Numpy 

"""

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    a = np.array(a)
    x = a.transpose()
    return x


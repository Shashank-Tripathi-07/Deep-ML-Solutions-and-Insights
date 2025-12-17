"""

48. Implement Reduced Row Echelon Form (RREF) Function [Medium] 

In this problem, your task is to implement a function that converts a given matrix into its Reduced Row Echelon Form (RREF). The RREF of a matrix is a special form where each leading entry in a row is 1, and all other elements in the column containing the leading 1 are zeros, except for the leading 1 itself.

However, there are some additional details to keep in mind:

Diagonal entries can be 0 if the matrix is reducible (i.e., the row corresponding to that position can be eliminated entirely).
Some rows may consist entirely of zeros.
If a column contains a pivot (a leading 1), all other entries in that column should be zero.
Your task is to implement the RREF algorithm, which must handle these cases and convert any given matrix into its RREF.

Example:
Input:
import numpy as np

matrix = np.array([
    [1, 2, -1, -4],
    [2, 3, -1, -11],
    [-2, 0, -3, 22]
])

rref_matrix = rref(matrix)
print(rref_matrix)
Output:
# array([
#    [ 1.  0.  0. -8.],
#    [ 0.  1.  0.  1.],
#    [-0. -0.  1. -2.]
# ])
Reasoning:
The given matrix is converted to its Reduced Row Echelon Form (RREF) where each leading entry is 1, and all other entries in the leading columns are zero.

INSIGHTS or Learnings: 

We perform RREF calculations to effectively break down complex looking equations and make them simple enough that we can understand the relations between them 
This may look like a small, head-breaking calculation for now but in the larger image when you're actually training a model, this actually becomes a major help in your work. 
RREF makes the structure visible in short, 

It also answers questions like is the system consistent/inconsistent, how many solutions exist and the basis of solution space, which brings certainity rather than pure intuition. 

Signal systems like Internet, calls and even radio signals use this a lot, highlighting how important it is. 

NOW SOLVE THE QUESITION YOURSELF and then check my solution also :) 

SOLUTION

I've tried multiple appraoches like Pivot Search and others but here I'll give you the Gauss-Jordan Elimination version as it very closely mimicks the textbook mathematics. 
"""

import numpy as np

def rref(matrix):
	A = np.array(matrix, dtype=float)
    rows, cols = A.shape

    pivot_col = 0

    for r in range(rows):
        if pivot_col >= cols:
            break

        # Find pivot in current column
        pivot_row = None
        for i in range(r, rows):
            if A[i, pivot_col] != 0:
                pivot_row = i
                break

        # Move right if column has no pivot
        while pivot_row is None:
            pivot_col += 1
            if pivot_col >= cols:
                return A
            for i in range(r, rows):
                if A[i, pivot_col] != 0:
                    pivot_row = i
                    break

        # Swap pivot row into position
        if pivot_row != r:
            A[[r, pivot_row]] = A[[pivot_row, r]]

        # Normalize pivot to 1
        A[r] = A[r] / A[r, pivot_col]

        # Eliminate other rows
        for i in range(rows):
            if i != r and A[i, pivot_col] != 0:
                A[i] -= A[i, pivot_col] * A[r]

        pivot_col += 1

    return A

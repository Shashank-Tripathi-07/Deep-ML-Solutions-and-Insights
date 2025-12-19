"""
201. QR Decomposition [Hard] 

Implement QR decomposition using the Gram-Schmidt process. 
Given a matrix A, decompose it into the product of an orthogonal matrix Q (where columns are orthonormal) and an upper triangular matrix R, 
such that A = Q @ R.
Return both Q and R as a tuple of matrices.

Example:
Input:
A = [[3, 0], [4, 5]]
Output:
Q = [[0.6, -0.8], [0.8, 0.6]], R = [[5.0, 4.0], [0.0, 3.0]]

Insights: 

- We do QR decomposition of matrices to reduce the possibility of them being similar or even same. This helps us reduce a lot of mathematical workload 
- Geometrically, QR answers the question: “How can we express the columns of A in an orthonormal coordinate system?”
-Gram–Schmidt is the algorithm that constructs an orthonormal basis from a set of linearly independent vectors.
-Used in solving least squares problems. 
- Modified versions of it are used in modern day libraries. 
- Think of QR like this: “Take a messy coordinate system and rotate it into a clean, orthogonal one — without changing the space it spans.”
"""

#SOLTION:

import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
    A = np.array(A, dtype=float)
    x, y = A.shape

    Q = np.zeros((x, y))
    R = np.zeros((x, y))

    for j in range(y):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            return -1  # linearly dependent columns

        Q[:, j] = v / R[j, j]

    return Q.tolist(), R.tolist()	







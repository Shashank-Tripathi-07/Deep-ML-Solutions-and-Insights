"""

12. Singular Value Decomposition (SVD) [Hard] 

Write a Python function called svd_2x2_singular_values(A) that finds an approximate singular value decomposition of a real 2 x 2 matrix using one Jacobi rotation. Input A: a NumPy array of shape (2, 2)

Rules You may use basic NumPy operations (matrix multiplication, transpose, element wise math, etc.). Do not call numpy.linalg.svd or any other high-level SVD routine. Stick to a single Jacobi step no iterative refinements.

Return A tuple (U, Î£, V_T) where U is a 2 x 2 orthogonal matrix, Î£ is a length 2 NumPy array containing the singular values, and V_T is the transpose of the right-singular-vector matrix V.

Example:
Input:
a = [[2, 1], [1, 2]]
Output:
(array([[-0.70710678, -0.70710678],
                        [-0.70710678,  0.70710678]]),
        array([3., 1.]),
        array([[-0.70710678, -0.70710678],
               [-0.70710678,  0.70710678]]))
Reasoning:
U is the first matrix sigma is the second vector and V is the third matrix


Insights: 

- The problem is a very good problem itself, the code implementation is longer than the other ones in Linear Algebra but teaches you a lot. 
- I've explained the steps in the solution. 
- I'd say take a look on the solution also and map out the maths to code of it. 

Solution: 

"""



import numpy as np

def svd_2x2_singular_values(A):
    # Convert input to float array
    A = np.array(A, dtype=float)
    
    # Step 1: Form the Gram matrix B = A^T @ A
    # This is a symmetric positive semi-definite matrix whose eigenvalues
    # are the squares of the singular values of A
    B = A.T @ A
    
    # Step 2: Compute Jacobi rotation angle to diagonalize B
    # For a symmetric 2x2 matrix [[a, b], [b, c]], the angle that zeros
    # the off-diagonal is given by: tan(2*theta) = 2*b / (a - c)
    # We use arctan2 for numerical stability
    if abs(B[0, 1]) > 1e-10:
        theta = 0.5 * np.arctan2(2 * B[0, 1], B[0, 0] - B[1, 1])
    else:
        theta = 0  # B is already diagonal
    
    # Step 3: Construct the Jacobi rotation matrix V
    # This is a 2D rotation matrix that will diagonalize B
    c, s = np.cos(theta), np.sin(theta)
    V = np.array([[c, -s], [s, c]])
    
    # Step 4: Diagonalize B by similarity transformation
    # D = V^T @ B @ V contains eigenvalues of B on the diagonal
    D = V.T @ B @ V
    
    # Step 5: Extract singular values as square roots of eigenvalues
    # Use maximum(0, ...) to handle numerical errors that might give tiny negative values
    sigma = np.sqrt(np.maximum(0, [D[0, 0], D[1, 1]]))
    
    # Step 6: Sort singular values in descending order (convention)
    # If needed, swap the order and corresponding columns of V
    if sigma[0] < sigma[1]:
        sigma = sigma[[1, 0]]
        V = V[:, [1, 0]]
    
    # Step 7: Compute left singular vectors U
    # From SVD: A = U @ diag(sigma) @ V^T, so U @ diag(sigma) = A @ V
    # Therefore: U[:, i] = (A @ V[:, i]) / sigma[i]
    U = np.column_stack([(A @ V[:, i]) / sigma[i] if sigma[i] > 1e-10 else [1, 0] for i in range(2)])
    
    # Step 8: Ensure U is orthonormal using Gram-Schmidt
    # Make second column orthogonal to first
    U[:, 1] -= np.dot(U[:, 1], U[:, 0]) * U[:, 0]
    # Normalize both columns
    U /= np.linalg.norm(U, axis=0)
    
    # Return (U, Sigma, V^T)
    return (U, sigma, V.T)

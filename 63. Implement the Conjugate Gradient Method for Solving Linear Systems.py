"""
63. Implement the Conjugate Gradient Method for Solving Linear Systems [Hard]

Task: Implement the Conjugate Gradient Method for Solving Linear Systems
Your task is to implement the Conjugate Gradient (CG) method, an efficient iterative algorithm for solving large, sparse, symmetric, positive-definite linear systems. 
Given a matrix A and a vector b, the algorithm will solve for x in the system ( Ax = b ).

Write a function conjugate_gradient(A, b, n, x0=None, tol=1e-8) that performs the Conjugate Gradient method as follows:

A: A symmetric, positive-definite matrix representing the linear system.
b: The vector on the right side of the equation.
n: Maximum number of iterations.
x0: Initial guess for the solution vector.
tol: Tolerance for stopping criteria.
The function should return the solution vector x.

Example:
Input:
A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
n = 5

print(conjugate_gradient(A, b, n))
Output:
[0.09090909, 0.63636364]
Reasoning:
The Conjugate Gradient method is applied to the linear system Ax = b with the given matrix A and vector b. The algorithm iteratively refines the solution to converge to the exact solution.


Insights and Analysis: 

Conjugate Gradient solves linear systems by walking straight to the solution without ever stepping backwards. This is the whole question's theory explained. 
Now we'll go on to the code and show you how I did it. 

"""
#SOLUTION: 

import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""

    # Initial guess
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0, dtype=float)

    # Initial residual and direction
    r = b - A @ x
    c = r.copy()

    for _ in range(n):
        Ac = A @ c

        alpha = (r @ r) / (c @ Ac)    # @ is actually matrix multiplication
        x = x + alpha * c
        r_new = r - alpha * Ac

        # Convergence check
        if np.linalg.norm(r_new) < tol:
            break

        beta = (r_new @ r_new) / (r @ r)
        c = r_new + beta * c
        r = r_new

    return x.tolist()

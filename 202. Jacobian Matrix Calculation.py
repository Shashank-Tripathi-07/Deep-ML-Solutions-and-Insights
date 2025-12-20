"""
202. Jacobian Matrix Calculation [Medium] 

Implement a function to compute the Jacobian matrix of a vector-valued function using numerical differentiation.
The Jacobian matrix contains all first-order partial derivatives of a function f: R^n -> R^m.
Given a function f and a point x, approximate each partial derivative using finite differences and return the m x n Jacobian matrix.

Example:

Input:
f(x, y) = [x^2, xy, y^2], x = [2, 3]

Output:
[[4.0, 0.0], [3.0, 2.0], [0.0, 6.0]]

Insights: 

- Revisit the jacobian matrix if you want, take a look at the philosophical perspective...what exactly is it doing ? 
- Yes, that's the whole algorithm you need to code
- and that's the solution 

"""

#My_SOLUTION: 

import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
    x = np.array(x, dtype=float)     #Initializing things
    f_x = np.array(f(x), dtype=float)

    m = f_x.size      # number of outputs
    n = x.size        # number of inputs

    J = np.zeros((m, n))           #Initializing 

    for j in range(n):                   #The real mathematical magic happens here
        x_step = x.copy()                # we need a copy because we don't want to change the real value
        x_step[j] += h

        f_step = np.array(f(x_step), dtype=float)           

        # finite difference for column j
        J[:, j] = (f_step - f_x) / h               #The exact maths formula but in code

    return J   

""" 
215. Partial Derivatives of Multivariable Functions [Medium]

Implement a function to compute partial derivatives of multivariable functions at a given point. Partial derivatives measure the rate of change with respect to one variable while holding others constant. Given a function name and a point, return the tuple of all partial derivatives at that point.

Example:
Input:
func_name='poly2d', point=(2.0, 3.0)
Output:
(21.0, 16.0)
Reasoning:
f(x,y) = x²y + xy². ∂f/∂x = 2xy + y² = 2(2)(3) + 9 = 21. ∂f/∂y = x² + 2xy = 4 + 2(2)(3) = 16. Gradient at (2,3) is (21, 16).


Insights: 

- This question is a good question, take a good look and then solve
- The input is in the form of string and you need to solve it numerically, hence we use if-elif-else to get the job done. 
- The question also teaches you very well about the floating-point precision in Python 
- The calculation for partial derivates is same as before (same maths-to-code approach) 
- I hope, you get the insight to solving this question. You can check my solution if you want. 

"""

#SOLUTION: 

import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
    h = 1e-8
    x = np.array(point, dtype=float)

    # Define the function based on func_name
    if func_name == 'poly2d':
        # f(x,y) = x^2 y + x y^2
        f = lambda v: v[0]**2 * v[1] + v[0] * v[1]**2

    elif func_name == 'exp_sum':
        # f(x,y) = e^(x+y)
        f = lambda v: np.exp(v[0] + v[1])

    elif func_name == 'product_sin':
        # f(x,y) = x * sin(y)
        f = lambda v: v[0] * np.sin(v[1])

    elif func_name == 'poly3d':
        # f(x,y,z) = x^2 y + y z^2
        f = lambda v: v[0]**2 * v[1] + v[1] * v[2]**2

    elif func_name == 'squared_error':
        # f(x,y) = (x - y)^2
        f = lambda v: (v[0] - v[1])**2

    else:
        return -1

    base_value = f(x)
    derivatives = []

    # Finite difference for each variable
    for i in range(len(x)):
        x_step = x.copy()
        x_step[i] += h
        derivative = (f(x_step) - base_value) / h
        derivatives.append(derivative)

    return tuple(derivatives)   #The question explicitly stated to return Tuple and hence, we convert the output to Tuple. 

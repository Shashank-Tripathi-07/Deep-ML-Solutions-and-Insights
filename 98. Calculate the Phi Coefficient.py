""" 

98. Calculate the Phi Coefficient [Easy] 

Implement a function to calculate the Phi coefficient, a measure of the correlation between two binary variables.
The function should take two lists of integers (0s and 1s) as input and return the Phi coefficient rounded to 4 decimal places.

Example:
Input:
phi_corr([1, 1, 0, 0], [0, 0, 1, 1])
Output:
-1.0
Reasoning:
The Phi coefficient measures the correlation between two binary variables. 
In this example, the variables have a perfect negative correlation, resulting in a Phi coefficient of -1.0.

Insights: 

- The Phi coefficient is a type of correlation coefficient , which is used when we need to find the correlation between two binary variables.
- ϕ= (x00+x01)(x10+x11)(x00+x10)(x01+x 11)(x00⋅x11)−(x01⋅x 10)

Explanation of Terms:
(x_00): The number of cases where (x = 0) and (y = 0).
(x_01): The number of cases where (x = 0) and (y = 1).
(x_10): The number of cases where (x = 1) and (y = 0).
(x_11): The number of cases where (x = 1) and (y = 1).

- Now get to the code and build your version, here's my solution: 
"""

#SOLUTION:

import numpy as np 

def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""

    x = np.array(x)
    y = np.array(y)


    # Contingency table counts
    a = np.sum((x == 1) & (y == 1))
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    d = np.sum((x == 0) & (y == 0))

    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d)) 
    if denom == 0:
        return 0.0

    phi = (a * d - b * c) / denom       #the phi coefficient formula 
    return round(phi, 4)



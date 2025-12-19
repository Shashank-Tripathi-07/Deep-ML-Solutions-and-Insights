"""
84. Phi Transformation for Polynomial Features [Easy] 

Write a Python function to perform a Phi Transformation that maps input features into a higher-dimensional space by generating polynomial features. 
The transformation allows models like linear regression to fit nonlinear data by introducing new feature dimensions that represent polynomial combinations of the original input features.
The function should take a list of numerical data and a degree as inputs, and return a nested list where each inner list represents the transformed features of a data point.
If the degree is less than 0, the function should return an empty list.

Example:
Input:
data = [1.0, 2.0], degree = 2
Output:
[[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]]
Reasoning:
The Phi Transformation generates polynomial features for each data point up to the specified degree. For data = [1.0, 2.0] and degree = 2, 
the transformation creates a nested list where each row contains powers of the data point from 0 to 2.


Insights: 

- All numbers go through Numpy, and then we think of anything else. 
- For every numeric calculaion, use Numpy, it's a better option than to get into a fight to explain python that you're using a list. 
"""


#SOLUTION: 

import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# returning empty list when degree is 0 
	if degree < 0: 
		return [] 

    # Convert input data to NumPy array
    data = np.asarray(data, dtype=float)

    # Generate powers from 0 to degree
    powers = np.arange(degree + 1)

    # Compute polynomial features using broadcasting
    phi = data[:, None] ** powers

    return phi.tolist()

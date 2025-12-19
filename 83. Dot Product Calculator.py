"""
83. Dot Product Calculator [Easy] 

Write a Python function to calculate the dot product of two vectors. The function should take two 1D NumPy arrays as input and return the dot product as a single number.

Example:
Input:
vec1 = np.array([1, 2, 3]), vec2 = np.array([4, 5, 6])
Output:
32

Reasoning:
The function calculates the dot product by multiplying corresponding elements of the two vectors and summing the results.
For vec1 = [1, 2, 3] and vec2 = [4, 5, 6], the result is (1 * 4) + (2 * 5) + (3 * 6) = 32.

Insights: 

-None, the question is easy and doable. 


"""

#SOLUTION: 

import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	"""
	return np.dot(vec1,vec2) 

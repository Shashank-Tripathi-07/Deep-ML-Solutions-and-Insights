"""

Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists. Additionally, provide test cases to verify the correctness of your implementation.

Example:
Input:
[[1, 2, 3], [4, 5, 6]]
Output:
[[1.0, 1.0], [1.0, 1.0]]

Reasoning:
The covariance between the two features is calculated based on their deviations from the mean. For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix.


INSIGHTS: 

- The question get's solved easily but the main thing to learn here is to take care of the input and output. 
- Always do numerical operations → rounding → .tolist() (in that order).

SOLUTION: 
""" 

import numpy as np 

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	
    x = np.array(vectors, dtype = float)

    result = np.cov(x).tolist()
    
	  return result 

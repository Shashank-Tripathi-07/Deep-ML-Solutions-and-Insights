"""

66.  Implement Orthogonal Projection of a Vector onto a Line [Easy] 

Task: Compute the Orthogonal Projection of a Vector
Your task is to implement a function that calculates the orthogonal projection of a vector v onto another vector L. This projection results in the vector on L that is closest to v.

Write a function orthogonal_projection(v, L) that takes in two lists, v (the vector to be projected) and L (the line vector), and returns the orthogonal projection of v onto L. The function should output a list representing the projection vector rounded to three decimal places.

Example:
Input:
v = [3, 4]
L = [1, 0]
print(orthogonal_projection(v, L))
Output:
[3.0, 0.0]

Reasoning:
The orthogonal projection of vector [3, 4] onto the line defined by [1, 0] results in the projection vector [3, 0], which lies on the line [1, 0].

Insights and Analysis: 

- You can calculate manually but don't cause you have numpy...be smarter not faster [remember the work smarter meme] 
- Yeah the question is easy. 
"""

#SOLUTION: 

import numpy as np 

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v = np.array(v, dtype=float)
	L = np.array(L, dtype=float)

	factor = np.dot(v,L)/np.dot(L,L)
	projection = factor * L 

	return np.round(projection,3).tolist()

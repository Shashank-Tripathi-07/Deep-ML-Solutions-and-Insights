"""

55. 2D Translation Matrix Implementation [Medium] 

Task: Implement a 2D Translation Matrix
Your task is to implement a function that applies a 2D translation matrix to a set of points.
A translation matrix is used to move points in 2D space by a specified distance in the x and y directions.
Write a function translate_object(points, tx, ty) where points is a list of [x, y] coordinates and tx and ty are the translation distances in the x and y directions, respectively.
The function should return a new list of points after applying the translation matrix.

Example:
Input:
points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))
Output:
[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]

Reasoning:
The translation matrix moves the points by 2 units in the x-direction and 3 units in the y-direction. The resulting points are [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]].

Approach, Analysis and Insights: 

- I took the approach of converting these two different set of inputs (i.e. points and translation_points) into two different numpy arrays and then adding them together with numpy operations. 
- Nothing else is required for this question, simple, sleek and elegant question :) 
"""

# SOLUTION:

import numpy as np
def translate_object(points, tx, ty):
	
	points = np.array(points, dtype=float)
	translation = np.array([tx,ty], dtype=float)

	translated_points = (points + translation).tolist()

	
	
	return translated_points

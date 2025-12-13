""" 
4. Calculate Mean by Row or Column [Easy] 

Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

Example:
Input:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
Output:
[4.0, 5.0, 6.0]

Reasoning:
Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].

INSIGHTS: 

#You can use Numpy's mean function that too in axis=1 for row-wise mean and axis=2 for column-wise mean. 
#If and elif clause in python can have conditions and not else clause. 

SOLUTION: 

"""

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

	import numpy as np
	
    arr = np.array(matrix) 

	if (mode == "row"): 
		return arr.mean(axis=1)

	elif (mode == "column"): 
		return arr.mean(axis=0) 


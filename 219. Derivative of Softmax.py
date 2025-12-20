"""

219. Derivative of Softmax [Medium] 

Write a Python function that computes the Jacobian matrix of the softmax function.
The softmax function maps a vector of real numbers to a probability distribution,
and its derivative is essential for backpropagation in neural networks with classification outputs.
Given an input vector x, return the Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j).

Example:
Input:
x = [1.0, 2.0, 3.0]
Output:
[[0.0819, -0.022, -0.0599], [-0.022, 0.1848, -0.1628], [-0.0599, -0.1628, 0.2227]]
Reasoning:
First compute softmax: s = [0.09, 0.2447, 0.6652]. Then for diagonal elements: J[i][i] = s[i] * (1 - s[i]).
For off-diagonal: J[i][j] = -s[i] * s[j].
For example, J[0][0] = 0.09 * (1 - 0.09) = 0.0819 and J[0][1] = -0.09 * 0.2447 = -0.022.

Insights: 

- You've to simply implement the maths-to-code. 
- Nothing else, a great question 

"""

#SOLUTION: 


import numpy as np 

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	x = np.array(x,dtype=float)


	#numerically stabilizing softmax
	exponent_x = np.exp(x-np.max(x))
	s = exponent_x / np.sum(exponent_x)

	n = len(s)
	J = np.zeros((n,n))


    # Converting the mathematical formula to code
	for i in range(n):
		for j in range(n):
			if i == j: 
				J[i,j] = s[i]*(1-s[i])

			else: 
				J[i,j] = -s[i] * s[j]

	return np.round(J,6).tolist()    #doing .round() to give the required answer and then .tolist() to avoid passing the whole array as a result. 


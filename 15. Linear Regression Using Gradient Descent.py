"""
15. Linear Regression Using Gradient Descent [Easy]

Write a Python function that performs linear regression using gradient descent.
The function should take NumPy arrays X (features with a column of ones for the intercept) 
and y (target) as input, along with learning rate alpha and the number of iterations, and
return the coefficients of the linear regression model as a NumPy array. 
Round your answer to four decimal places. 
-0.0 is a valid result for rounding a very small number.

Example:
Input:
X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
Output:
np.array([0.1107, 0.9513])
Reasoning:
The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.

Insights: 

- Here, we follow the same approach as mathematics, 
  -understand the maths
  - think of how to convert it into code
  - write the code
  - check weather what you've written is valid
  - click on submit button to see if the problem builder agrees with your solution
  - yeah, if every test is green then, you're a genius :) 

  -Now let's take a look at how I calculated it....If you've still failed with your solution, go from the first principles, code your solution and try again...

  """

#SOLUTION: 

import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	"""
	Perform linear regression using gradient descent.

	m = number of training examples
	n = number of parameters (features), technically n-1 features, 1st column is for intercept

	X: shape (m, n), `m` training examples with `n` input values for each feature
	y: shape (m, 1) array with the target values (ground truth)
	alpha: learning rate
	iterations: number of gradient descent steps
	"""

	m, n = X.shape
	y = y.reshape(-1, 1) 	# Make sure y is a column vector
	theta = np.zeros((n, 1))


    #Here I've converted the maths into code. 
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta -= alpha * gradient

	return np.round(theta.flatten(), 4) 	# Rounded to 4 decimals

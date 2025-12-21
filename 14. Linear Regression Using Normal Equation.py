"""
14. Linear Regression Using Normal Equation [Easy] 

Write a Python function that performs linear regression using the normal equation. 
The function should take a matrix X (features) and a vector y (target) as input, and 
return the coefficients of the linear regression model. Round your answer to four decimal places, 
-0.0 is a valid result for rounding a very small number.

Example:
Input:
X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
Output:
[0.0, 1.0]

Reasoning:
The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.

Insights: 

- We'll calculate the theta through the matrix formula θ=(X TX)^−1*(X^T)*y
  - y is the vector of target values, X is the input matrix and transpose is represented by T and ^-1 stands for inverse. 

- We calculate the theta that we put in the linear regression equation y = mx+c to calculate the value of y. 

"""
#SOLUTION:

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	
    X_new = np.array(X, dtype=float)
    y_new = np.array(y, dtype=float)

    #calculating theta that we can put as m in y = mx 
    m = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y_new

	return [round(v,4) for v in m.tolist()]

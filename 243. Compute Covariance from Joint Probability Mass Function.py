"""

Given two discrete random variables X and Y with their possible values and joint probability mass function (PMF), compute the covariance between X and Y.

The covariance measures the linear relationship between two random variables.
A positive covariance indicates that the variables tend to increase together, while a negative covariance indicates that when one increases, the other tends to decrease.

Your Task: Write a function covariance_from_joint_pmf(x_values, y_values, joint_pmf) that takes:

x_values: A list of possible values for random variable X
y_values: A list of possible values for random variable Y
joint_pmf: A 2D numpy array where joint_pmf[i][j] represents P(X=x_values[i], Y=y_values[j])
The function should return the covariance Cov(X, Y) as a float.

Note: You will need to compute marginal probabilities from the joint PMF and then calculate the expected values needed for the covariance formula.

Example:
Input:
x_values = [0, 1], y_values = [0, 1], joint_pmf = [[0.4, 0.1], [0.1, 0.4]]
Output:
0.15
Reasoning:
First, compute marginal probabilities: P(X=0)=0.5, P(X=1)=0.5, P(Y=0)=0.5, P(Y=1)=0.5.
Then E[X]=0.5, E[Y]=0.5. E[XY] = 000.4 + 010.1 + 100.1 + 110.4 = 0.4. Covariance = E[XY] - E[X]*E[Y] = 0.4 - 0.25 = 0.15. 
The positive covariance indicates X and Y tend to increase together.

Insights: 

- We are given complete probabilistic information about two random variables X and Y in the form of a joint probability mass function.
- From that, we want to compute a single number — the covariance — that answers this question:
- When X is larger than usual, is Y also larger than usual (or smaller)? And by how much, on average?

- Now, code yourself cause we've done a lot of covariance questions before this. If you get stuck, you can see my code. 

"""

#SOLUTION: 

import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """

    # Initializing arrays from given data 
    x_values = np.array(x_values, dtype=float)
    y_values = np.array(y_values, dtype=float)
    joint_pmf = np.array(joint_pmf, dtype=float)

    # Marginal probabilities
    p_x = joint_pmf.sum(axis=1)   # P(X)
    p_y = joint_pmf.sum(axis=0)   # P(Y)

    # Expected values of X and Y 
    E_X = np.sum(x_values * p_x)
    E_Y = np.sum(y_values * p_y)

    # Expected value of XY
    E_XY = 0.0
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            E_XY += x_values[i] * y_values[j] * joint_pmf[i, j]

    # Covariance
    return float(E_XY - E_X * E_Y)

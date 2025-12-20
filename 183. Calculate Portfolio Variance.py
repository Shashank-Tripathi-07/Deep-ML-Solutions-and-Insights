"""

183. Calculate Portfolio Variance [Easy] 

Write a Python function to calculate the variance of a portfolio given a covariance matrix of asset returns and a set of portfolio weights.
The function should ensure that the inputs are valid and compatible, and return a single floating-point number representing the variance.

Example:
Input:
cov_matrix = [[0.1, 0.02], [0.02, 0.15]], weights = [0.6, 0.4]
Output:
0.0696

Reasoning:
The covariance between the two assets reduces the overall variance compared to a simple weighted average of individual variances.

Insights: 

- We need to find the variance of the whole portfolio, given the covariance matrix of asset returns and set of portfolio weights. 
- We can solve this by using Numpy's covariance function and using the mathemtical formula, w^T * covariance matrix * weights. 

"""

#SOLUTION: 

import numpy as np

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio.

    Args:
        cov_matrix (list[list[float]]): Covariance matrix of asset returns.
        weights (list[float]): Portfolio weights.

    Returns:
        float: Portfolio variance.
    """
    cov = np.array(cov_matrix, dtype=float)
    w = np.array(weights, dtype=float)

    # Validate dimensions
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return -1
    if cov.shape[0] != w.shape[0]:
        return -1

    # Portfolio variance: w^T Σ w
    variance = w.T @ cov @ w

    return float(variance)

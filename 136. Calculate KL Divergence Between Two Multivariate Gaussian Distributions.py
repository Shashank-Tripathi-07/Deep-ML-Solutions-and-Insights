"""

KL divergence measures the dissimilarity between two probability distributions.
In this problem, you'll implement a function to compute the KL divergence between two multivariate Gaussian distributions given their means and covariance matrices.
Use the provided mathematical formulas and numerical considerations to ensure accuracy.

Example:
Input:
mu_p, Cov_p, mu_q, Cov_q for two random multivariate Gaussians
Output:
A float representing the KL divergence
Reasoning:
The KL divergence is calculated using the formula: 0.5 * (log det term, minus dimension p,
Mahalanobis distance between means, and trace term). It measures how dissimilar the second Gaussian is from the first.

Insights: 

- same maths to code case but I'd like you to read the maths and formula. It's something I can't teach here as it'll take a lot of time. Nevertheless, I'll be writing comments in my solution to give you a hint. 

"""




#SOLUTION: 


import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Dimensionality
    p = mu_p.shape[0]

    # Inverse and determinant of Cov_q
    Cov_q_inv = np.linalg.inv(Cov_q)

    # Log-determinant term (numerically stable)
    log_det_ratio = np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p))

    # Trace term
    trace_term = np.trace(Cov_q_inv @ Cov_p)

    # Mean difference (Mahalanobis distance term)
    diff = mu_p - mu_q
    mahalanobis_term = diff.T @ Cov_q_inv @ diff

    # KL divergence formula
    kl_div = 0.5 * (log_det_ratio - p + trace_term + mahalanobis_term)

    return float(kl_div)

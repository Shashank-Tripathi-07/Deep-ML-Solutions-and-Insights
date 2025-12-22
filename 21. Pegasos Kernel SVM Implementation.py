"""
21. Pegasos Kernel SVM Implementation [Hard] 

Write a Python function that implements a deterministic version of the Pegasos algorithm to train a kernel SVM classifier from scratch.
The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature),
a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF),
regularization parameter (lambda), and the number of iterations. Note that while
the original Pegasos algorithm is stochastic (it selects a single random sample at each step),
this problem requires using all samples in every iteration (i.e., no random sampling).
The function should perform binary classification and return the model's alpha coefficients and bias.

Example:
Input:
data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'rbf', lambda_val = 0.01, iterations = 100, sigma = 1.0
Output:
alpha = [0.03, 0.02, 0.05, 0.01], b = -0.05
Reasoning:
Using the RBF kernel, the Pegasos algorithm iteratively updates the weights based on a sub-gradient descent method,
taking into account the non-linear separability of the data induced by the kernel transformation.

Insights: 

- Pegasos stands for Primal Estimated sub-GrAdient SOlver for SVM. It's an efficient algorithm for training Support Vector Machines (SVMs), particularly useful for large-scale problems.
- Pegasos is used in case of building SVMs for large-scale data. 
- We're doing a simpler implementation of pegasos in this question. 

"""

#SOLUTION: 

import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100,sigma=1.0) -> (list, float):
    
    n = data.shape[0]
    y = labels.astype(float)

    # ---------- Kernel matrix ----------
    if kernel == "linear":
        K = data @ data.T

    elif kernel == "rbf":
        sq_norms = np.sum(data ** 2, axis=1)
        K = sq_norms[:, None] + sq_norms[None, :] - 2 * data @ data.T
        K = np.exp(-K / (2 * sigma ** 2))

    else:
        raise ValueError("Kernel must be 'linear' or 'rbf'")

    # ---------- Initialization ----------
    alphas = np.zeros(n)
    b = 0.0

    # ---------- Training loop ----------
    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)

        for i in range(n):
            decision = np.sum(alphas * y * K[:, i]) + b
            margin = y[i] * decision

            if margin < 1:
                alphas[i] += eta * (y[i] - lambda_val * alphas[i])
                b += eta * y[i]

    return alphas.tolist(), float(b)

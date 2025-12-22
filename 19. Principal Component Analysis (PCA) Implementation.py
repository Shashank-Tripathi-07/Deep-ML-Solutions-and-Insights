"""
19. Principal Component Analysis (PCA) Implementation [Medium]

Write a Python function that performs Principal Component Analysis (PCA) from scratch.
The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature.
The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors,
and return the principal components (the eigenvectors corresponding to the largest eigenvalues).
The function should also take an integer k as input,
representing the number of principal components to return.

Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
Output:
[[0.7071], [0.7071]]
Reasoning:
After standardizing the data and computing the covariance matrix,
the eigenvalues and eigenvectors are calculated.
The largest eigenvalue's corresponding eigenvector is returned as the principal component,
rounded to four decimal places.

Insights: 

-Keep checking what is happening to the data at every step do a lot of .shape() and checks to see if you're performing calculations on the right row or columnn 

"""

#SOLUTION:


import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	
    #initializing things
    X = np.array(data, dtype=float)

    #Standardizing the data with zero mean and unit variance
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1) 
    #ddof=1 is for sample standard deviation,also to keep the data same and #good. 

    X_std=(X-mean)/std

    #Calculating Covariance Matrix 
    cov_matrix=np.cov(X_std, rowvar=False)
    #rowvar=False to calculate covariance for features and not data. 
    

    eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)

    #Sorting eigenvalues and eigenvectors to return the most imp. values

    sor=np.argsort(eigenvalues)[::-1]  #descending order
    eigenvectors = eigenvectors[:,sor]

    #selecting the top k eigenvectors 
    principal_components=eigenvectors[:,:k] 


	return np.round(principal_components, 4)

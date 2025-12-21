"""
16. Feature Scaling Implementation [Easy] 

Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization.
The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature.
It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization.
Make sure all results are rounded to the nearest 4th decimal.

Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]])
Output:
([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
Reasoning:
Standardization rescales the feature to have a mean of 0 and a standard deviation of 1. Min-max normalization rescales the feature to a range 


Insights: 

- Just take the formula and implement it as code. 
- Don't forget to import Numpy
-Remember the formulas for later use

"""

#SOLUTION: 

import numpy as np 

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	
    x = np.asarray(data, dtype=float)

    mean = x.mean(axis=0)
    std = x.std(axis=0)

    standardized_data = (x-mean)/std

    #Min-Max normalization

    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    normalized_data = (x-x_min)/(x_max-x_min)

    return np.round(standardized_data, 4), np.round(normalized_data, 4)

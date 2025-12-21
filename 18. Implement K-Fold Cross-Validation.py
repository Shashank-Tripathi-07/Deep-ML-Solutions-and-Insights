"""
18. Implement K-Fold Cross-Validation [Medium]

Implement a function to generate train and test splits for K-Fold Cross-Validation.
Your task is to divide the dataset into k folds and return a list of train-test indices for each fold.

Example:
Input:
k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False)
Output:
[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
Reasoning:
The function splits the dataset into 5 folds without shuffling and returns train-test splits for each iteration.


Insights: 

-None, please check the code and understand what happens in case you don't understand theory or go wrong in the math-to-code implementation. 

"""

#SOLUTION: 

import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True,random_state=None):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    
    m = len(X)
    X_indices = np.arange(m) 


    if shuffle: 
       np.random.shuffle(X_indices)   

    #Not implementing the rng from np so as to handle the case of global injection of np.random.seed value.


    fold_sizes = [m//k]*k 
    #building k containers of size n//k 
    for i in range(m % k):
        fold_sizes[i] += 1
    #putting in any remaining value of n. 
    #did this to conserve unirformity in above calculation

    #Building the k-folds
    folds = []
    current_fold = 0 
    for fold_size in fold_sizes: 
        test_val = X_indices[current_fold:current_fold+fold_size]
        train_val = np.concatenate((X_indices[:current_fold], X_indices[current_fold+fold_size:]))

        folds.append((train_val.tolist(), test_val.tolist()))
        current_fold += fold_size



    return folds


    

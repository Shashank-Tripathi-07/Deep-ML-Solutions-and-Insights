"""
220. Derivative of Cross-Entropy Loss w.r.t. Logits [Medium] 

Write a Python function that computes the derivative (gradient) of the cross-entropy loss with respect to the input logits.
Given a vector of logits (raw model outputs before softmax) and a target class index, return the gradient vector.
This gradient is fundamental for training neural network classifiers and has an elegant closed-form solution.

Example:
Input:
logits = [1.0, 2.0, 3.0], target = 0
Output:
[-0.91, 0.2447, 0.6652]

Reasoning:
First compute softmax: p = [0.09, 0.2447, 0.6652]. The one-hot target vector is y = [1, 0, 0].
The gradient is simply p - y = [0.09 - 1, 0.2447 - 0, 0.6652 - 0] = [-0.91, 0.2447, 0.6652].
The negative gradient for class 0 indicates we should increase that logit to reduce loss.

Insights: 

-Just converting Maths-to-code 
-We convert this to exactly get the Loss w.r.t Logits so as to fine tune the system better and reduce the loss of the model. 

"""

#SOLUTION: 


import numpy as np 

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
    logits = np.array(logits, dtype=float)

    # Numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # One-hot target
    y = np.zeros_like(probs)
    y[target] = 1.0

    # Gradient
    grad = probs - y

    return grad


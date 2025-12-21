"""
120. Bhattacharyya Distance Between Two Distributions [Easy]

Implement a function to calculate the Bhattacharyya distance between two probability distributions.
The function should take two lists representing discrete probability distributions p and q, and return the Bhattacharyya distance rounded to 4 decimal places.
If the inputs have different lengths or are empty, return 0.0.

Example:
Input:
p = [0.1, 0.2, 0.3, 0.4], q = [0.4, 0.3, 0.2, 0.1]
Output:
0.1166
Reasoning:
The Bhattacharyya coefficient is calculated as the sum of element-wise square roots of the product of p and q, giving BC = 0.8898. The distance is then -log(0.8898) = 0.1166.

Insights: 

- Bhattacharyya Distance (BD) is a concept in statistics used to measure the similarity or overlap between two probability distributions P(x) and Q(x) on the same domain x.
- This differs from KL Divergence, which measures the loss of information when projecting one probability distribution onto another (reference distribution).

- Bhattacharyya Distance Formula : BC(P,Q)=∑ P(X)⋅Q(X)
                                   BD(P,Q)=−ln(BC(P,Q))
                                   where BC (P, Q) is the Bhattacharyya coefficient.


"""

#SOLUTIONS: 

import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Validate input
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return 0.0

    # Ensure valid probability distributions
    if np.any(p < 0) or np.any(q < 0):
        return 0.0

    # Normalize (safe even if already normalized)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0

    p /= p_sum
    q /= q_sum

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))

    if bc <= 0:
        return 0.0

    # Bhattacharyya distance
    distance = -np.log(bc)
    return round(distance, 4)

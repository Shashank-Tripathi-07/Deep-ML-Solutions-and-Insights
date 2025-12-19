"""
117. Compute Orthonormal Basis for 2D Vectors [Medium] 

Implement a function that computes an orthonormal basis for the subspace spanned by a list of 2D vectors using the Gram-Schmidt process.
The function should take a list of 2D vectors and a tolerance value (tol) to determine linear independence,
returning a list of orthonormal vectors (unit length and orthogonal to each other) that span the same subspace.
This is a fundamental concept in linear algebra with applications in machine learning, such as feature orthogonalization.

Example:
Input:
orthonormal_basis([[1, 0], [1, 1]])
Output:
[array([1., 0.]), array([0., 1.])]
Reasoning:
Start with [1, 0], normalize to [1, 0]. For [1, 1], subtract its projection onto [1, 0] (which is [1, 0]), leaving [0, 1].
Check if norm > 1e-10 (it is 1), then normalize to [0, 1].
The result is an orthonormal basis.



Insights: 

- This is a good calculation I've faced before in University maths and hence, was easy. 
- If you've studied Linear Algebra well then you'll understand this too
"""


#SOLUTION: 

import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis = []

    for v in vectors: 
        v = np.array(v,dtype=float)
 
        # Subtract projections onto existing basis vectors
        for u in basis: 
            v = v - np.dot(v,u)*u
        
        # Check if the remaining vector is non-zero (linearly independent)
        norm = np.linalg.norm(v)

        if norm > tol: 
            # Normalize and add to basis
            basis.append(v/norm)


    return basis

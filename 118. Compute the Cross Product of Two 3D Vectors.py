"""
118. Compute the Cross Product of Two 3D Vectors [Easy] 

Implement a function to compute the cross product of two 3-dimensional vectors. 
The cross product of two vectors results in a third vector that is perpendicular to both and follows the right-hand rule.
This concept is fundamental in physics, engineering, and 3D graphics.

Example:
Input:
cross_product([1, 0, 0], [0, 1, 0])
Output:
[0, 0, 1]
Reasoning:
The cross product of two orthogonal unit vectors [1, 0, 0] and [0, 1, 0] is [0, 0, 1], pointing in the positive z-direction as per the right-hand rule.


Insights: 

-None, a basic question showing the beauty of mathematics 
"""

#SOLUTION: 

import numpy as np

def cross_product(a, b):
    
    A = np.array(a, dtype=int)
    B = np.array(b, dtype=int)

    return np.cross(A,B)

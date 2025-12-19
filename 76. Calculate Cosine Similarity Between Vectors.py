"""
76. Calculate Cosine Similarity Between Vectors [Easy]

Task: Implement Cosine Similarity
In this task, you need to implement a function cosine_similarity(v1, v2) that calculates the cosine similarity between two vectors. Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity.

Input:
v1 and v2: Numpy arrays representing the input vectors.
Output:
A float representing the cosine similarity, rounded to three decimal places.

Constraints:
Both input vectors must have the same shape.
Input vectors cannot be empty or have zero magnitude.

Example:
Input:

import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(cosine_similarity(v1, v2))
Output:
1.0

Reasoning:
The cosine similarity between v1 and v2 is 1.0, indicating perfect similarity.

Insights: 

-None, the question is basic and can be done directly, but it's always good to handle edge cases :)
"""




#SOLUTION: 

import numpy as np

def cosine_similarity(v1, v2):
	# Compute dot product
    dot = np.dot(v1, v2)

    # Compute magnitudes (L2 norms)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Cosine similarity formula
    cos_sim = dot / (norm_v1 * norm_v2)

    # Round to three decimal places
    return round(float(cos_sim), 3)


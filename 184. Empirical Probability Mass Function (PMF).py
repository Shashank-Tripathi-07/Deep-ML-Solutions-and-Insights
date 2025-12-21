"""

184. Empirical Probability Mass Function (PMF) [Easy]

Problem
Given a list of integer samples drawn from a discrete distribution, 
implement a function to compute the empirical Probability Mass Function (PMF).
The function should return a list of (value, probability) pairs sorted by the value in ascending order.
If the input is empty, return an empty list.

Example:
Input:
samples = [1, 2, 2, 3, 3, 3]
Output:
[(1, 0.16666666666666666), (2, 0.3333333333333333), (3, 0.5)]

Reasoning:
Counts are {1:1, 2:2, 3:3} over 6 samples, so probabilities are 1/6, 2/6, and 3/6 respectively, returned sorted by value.

Insights: 

- We count the total length of all values and then find the occurence of each value and it's probability and then sort it before returning. 

"""

from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    if not samples:
        return []

    n = len(samples)
    counts = Counter(samples)

    return [(value, counts[value] / n) for value in sorted(counts)]
-Check the code for the implementation of this. 

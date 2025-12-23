"""

79. Binomial Distribution Probability [Medium]

Write a Python function to calculate the probability of achieving exactly k successes in
n independent Bernoulli trials, each with probability p of success, using the Binomial distribution formula.

Example:
Input:
n = 6, k = 2, p = 0.5
Output:
0.23438
Reasoning:
We want the probability of getting exactly 2 successes in 6 trials with 50% success rate.
The binomial coefficient C(6,2) = 15, and the probability calculation gives 15 × 0.25 × 0.0625 = 0.23438.

Insight: 

- We're just converting the formula P(X=k)=( n,k )⋅p^k⋅(1−p)^n−k
- Use math.comb to do the combinatrics calculation. 

"""

#SOLUTION: 



import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    if k < 0 or k > n or p < 0 or p > 1:
        return 0.0

    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))   

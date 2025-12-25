"""
Calculate the probability of occurrence of x in a Chi-Squared distribution with the given sample value 'x' and degrees of freedom 'k'.

Example:
Input:
x = 2, k = 2
Output:
0.184
Reasoning:
To calculate the probability density for x in a chi-square distribution with k degrees of freedom, we use the formula:

f(x; k) = (1 / (2^(k/2) * Gamma(k/2))) * x^((k/2) - 1) * exp(-x/2)

Substituting the values:

f(2; 2) = (1 / (2^1 * Gamma(1))) * 2^(1 - 1) * exp(-2/2) = (1 / (2 * 1)) * 1 * exp(-1) = 0.5 * 0.3679 ≈ 0.184

So, the probability density at x = 2 for k = 2 is approximately 0.184.


Insights: 

-The formula and maths are given in the question window. I hope you've arrived from the site so refer there for the theory, check the solution here if you get stuck

"""

#SOLUTION: 

import math

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    coeff = 1 / (2 ** (k / 2) * math.gamma(k / 2))
    val = coeff * (x ** (k / 2 - 1)) * math.exp(-x / 2)
    return round(val, 3)

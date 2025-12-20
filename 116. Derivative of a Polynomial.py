"""
116. Derivative of a Polynomial [Easy] 

Implement a function that computes the derivative of a polynomial term of the form c * x^n at a given point x, where c is a coefficient and n is the exponent. The function should return the value of the derivative, accounting for the coefficient in the power rule. This is useful for understanding how polynomials change at specific points in machine learning optimization problems.

Example:

Input:
poly_term_derivative(2.0, 3.0, 2.0)
Output:
12.0

Reasoning:
For the term 2 * x^2, the derivative is 2 * 2 * x^(2-1) = 4 * x. At x = 3, this evaluates to 4 * 3 = 12.0.

"""

#SOLUTION: 

def poly_term_derivative(c: float, x: float, n: float) -> float:
    
    if n == 0:
        return 0.0 

    return c*n*(x**(n-1))

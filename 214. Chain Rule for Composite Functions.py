"""
214. Chain Rule for Composite Functions [Medium]

Implement a function to compute the derivative of composite functions using the chain rule.
Given a list of functions (applied right to left) and a point x, calculate the derivative at that point.
Available functions: 'square' (xÂ²), 'sin', 'exp', 'log'. 
The chain rule states that for h(x) = f(g(x)), the derivative is h'(x) = f'(g(x)) Â· g'(x).

Example:
Input:
functions=['sin', 'square'], x=1.0
Output:
1.080605
Reasoning:
For h(x) = sin(x²): Inner function g(x) = x², outer f(u) = sin(u). At x=1: g(1)=1, g'(x)=2x so g'(1)=2. f'(u)=cos(u)
so f'(1)=cos(1)≈0.540. Chain rule: h'(1) = f'(g(1))·g'(1) = cos(1)·2 ≈ 1.0806.

Insights: 

- I've directly preferred the simple mathematics-to-code approach rather than a complex one. 
- You can use NumPy also to solve this. 

"""

#SOLUTION: 

import math

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
    value = x          # current value after applying functions
    derivative = 1.0   # running product of derivatives

    # Apply functions from right to left
    for fn in reversed(functions):

        if fn == 'square':
            derivative *= 2 * value
            value = value ** 2

        elif fn == 'sin':
            derivative *= math.cos(value)
            value = math.sin(value)

        elif fn == 'exp':
            derivative *= math.exp(value)
            value = math.exp(value)

        elif fn == 'log':
            derivative *= 1 / value
            value = math.log(value)

        else:
            return -1  # unsupported function

    return round(derivative, 6)

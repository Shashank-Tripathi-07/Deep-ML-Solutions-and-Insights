"""
127. Find Captain Redbeard's Hidden Treasure [Medium]

Captain Redbeard, the most daring pirate of the seven seas, has uncovered a mysterious ancient map. Instead of islands, it shows a strange wavy curve, and the treasure lies at the lowest point of the land! (watch out for those tricky local mins)

The land's height at any point 
x
x is given by:

f(x) = x^4 - 3x^3 + 2

Your Mission: Implement a Python function that finds the value of x where 
f(x) reaches its minimum, starting from any random initial position.

Example:
Input:
start_x = 0.0
Output:
min float value

Insights:

- This question doesn't get solve by the simplest answer approach. 
- You have to try a few times, HINT: Use NumPy 

"""

#SOLUTION: 

import numpy as np

def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Returns:
        float: The x-coordinate of the minimum point.
    """
    # Define the function
    def f(x):
        return x**4 - 3*x**3 + 2

    # Search over a wide range
    x_vals = np.linspace(-5, 5, 10000)
    y_vals = f(x_vals)

    # Find x where f(x) is minimum
    min_index = np.argmin(y_vals)
    return x_vals[min_index]

""" Rather than finding the best solution manually and converging it as the answer. We run the algorithm over all values, store them and then declare the minimum value of these as the asnwer. 
    and that's how Mr. Pirate get's the answer.... """


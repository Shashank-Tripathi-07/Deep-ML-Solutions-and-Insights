"""
Write a Python function to calculate the probability density function (PDF) of the normal distribution for a given value,
mean, and standard deviation. The function should use the mathematical formula of the normal distribution to return the PDF value rounded to 5 decimal places.

Example:
Input:
x = 16, mean = 15, std_dev = 2.04
Output:
0.17342
Reasoning:
The function computes the PDF using x = 16, mean = 15, and std_dev = 2.04.

Insights: 

-Consider reading theory for this question, I had to read it too


"""

#SOLUTION: 



import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
    # Standard deviation must be positive
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")

    #Calculating the normalization constant
    # Formula: 1 / (σ * sqrt(2π))
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))

    # Calculate the exponent part of the formula
    # Formula: - (x - μ)² / (2σ²)
    exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)

    # Combinining both parts using e^(exponent)
    val = coefficient * math.exp(exponent)

    # Step 4: Round the result to 5 decimal places
    return round(val, 5)

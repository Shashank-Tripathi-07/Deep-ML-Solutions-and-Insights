""" 
78. Descriptive Statistics Calculator  [Easy]

Write a Python function to calculate
various descriptive statistics metrics for a given dataset.
The function should take a list or NumPy array of numerical values and return a dictionary containing:

mean: Average of all values
median: Middle value when sorted
mode: Most frequently occurring value
variance: Population variance (divide by N)
standard_deviation: Square root of variance
25th_percentile, 50th_percentile, 75th_percentile: Quartile values
interquartile_range: Difference between 75th and 25th percentiles (IQR)
Example:
Input:
[1, 2, 2, 3, 4, 4, 4, 5]
Output:
{'mean': 3.125, 'median': 3.5, 'mode': 4, 'variance': 1.6094, 'standard_deviation': 1.2686, ...}
Reasoning:
Mean = (1+2+2+3+4+4+4+5)/8 = 3.125. Median = average of 4th and 5th values = (3+4)/2 = 3.5. Mode = 4 
(appears 3 times, most frequent). Variance and standard deviation measure spread around the mean.
Percentiles divide the sorted data into quarters

Insights: 
- we can do majority of these calculations with Numpy except mode. 
- try using maths where you get stuck with the code, it helps...

"""

#SOLUTIONS: 

import numpy as np
from collections import Counter
from math import sqrt 

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Convert input to NumPy array
    x = np.array(data, dtype=float)
    n = x.size

    if n == 0:
        return {}

    # Mean
    mean = x.mean()

    # Median
    median = np.median(x)

    # Mode (smallest value in case of tie)
    counts = Counter(x)
    max_freq = max(counts.values())
    mode = min(k for k, v in counts.items() if v == max_freq)

    # Population variance (divide by N)
    variance = ((x - mean) ** 2).mean()

    # Standard deviation
    standard_deviation = sqrt(variance)

    # Percentiles
    p25, p50, p75 = np.percentile(x, [25, 50, 75])

    # Interquartile Range
    iqr = p75 - p25

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "mode": int(mode),                      
        "variance": round(variance, 4),
        "standard_deviation": round(standard_deviation, 4),
        "25th_percentile": round(p25, 4),
        "50th_percentile": round(p50, 4),
        "75th_percentile": round(p75, 4),
        "interquartile_range": round(iqr, 4)
    }

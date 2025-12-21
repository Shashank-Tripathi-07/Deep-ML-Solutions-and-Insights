"""
212. Confidence Interval for Population Mean [Medium] 

Implement a function to calculate a confidence interval for a population mean using the t-distribution.
Given sample data and a confidence level (e.g., 0.95 for 95%), compute the sample mean, standard error,
margin of error, and the lower and upper bounds of the confidence interval.
The function should return a dictionary with all relevant statistics.

Example:
Input:
data=[10, 12, 11, 13, 14, 10, 12, 11], confidence_level=0.95
Output:
{'mean': 11.625, 'standard_error': 0.4978, 'margin_of_error': 1.177, 'lower_bound': 10.448, 'upper_bound': 12.802, 'confidence_level': 0.95}

Reasoning:
n=8, mean=11.625, s=1.408. SE = 1.408/√8 = 0.498. With df=7 and 95% confidence, t-critical = 2.365. 
ME = 2.365 × 0.498 = 1.177. CI = [11.625 - 1.177, 11.625 + 1.177] = [10.448, 12.802].
We are 95% confident the true mean is in this range.

Insights: 

- What Does "95% Confident" Mean?
Common misconception: "There's a 95% chance the true mean is in this interval."

Correct interpretation: "If we repeated this sampling procedure many times, 95% of the resulting confidence intervals would contain the true population mean."
The true mean is fixed (but unknown). The confidence interval is random (it varies with each sample). Our confidence is in the procedure, not any single interval.

"""

#SOLUTION:

import numpy as np
from scipy.stats import t

def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
	"""
	Calculate confidence interval for population mean.
	
	Args:
		data: Sample data
		confidence_level: Confidence level (default 0.95)
	
	Returns:
		Dictionary containing:
		- mean: Sample mean (point estimate)
		- standard_error: Standard error of the mean
		- margin_of_error: Margin of error
		- lower_bound: Lower bound of CI
		- upper_bound: Upper bound of CI
		- confidence_level: Confidence level used
	"""
    x = np.asarray(data, dtype=float)
    n = x.size

    if n < 2 or not (0 < confidence_level < 1):
        return {}

    # Sample statistics
    mean = x.mean()
    sample_std = x.std(ddof=1)

    # Standard error calculation
    standard_error = sample_std / np.sqrt(n)

    # Degrees of freedom
    df = n - 1

    # t critical value
    alpha = 1 - confidence_level
    t_crit = t.ppf(1 - alpha / 2, df)

    # Margin of error calculation
    margin_of_error = t_crit * standard_error

    # Confidence interval calculation
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return {
        "mean": round(mean, 4),
        "standard_error": round(standard_error, 4),
        "margin_of_error": round(margin_of_error, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "confidence_level": confidence_level
    }

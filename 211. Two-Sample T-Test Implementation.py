"""
211. Two-Sample T-Test Implementation [Medium] 

Implement a two-sample independent t-test (Welch's t-test) 
to determine if two samples have significantly different means.
The test should compute the t-statistic, p-value using the Welch-Satterthwaite degrees of freedom,
make a decision to reject or fail to reject the null hypothesis, and calculate Cohen's d effect size.
Welch's t-test does not assume equal variances between groups. Given two samples and a significance level alpha,
return a dictionary with the test results.

Example:
Input:
sample1=[12, 14, 13, 15, 14], sample2=[8, 9, 10, 9, 11], alpha=0.05
Output:
{'t_statistic': 5.8244, 'p_value': 0.000394, 'degrees_of_freedom': 8.0, 'reject_null': True, 'cohens_d': 3.6836}
Reasoning:
Mean₁=13.6, Mean₂=9.4, Var₁=1.3, Var₂=1.3. SE = sqrt(1.3/5 + 1.3/5) = 0.721. t = (13.6-9.4)/0.721 = 5.824. df = 8.0.
Two-tailed p = 0.000394. Since p < 0.05, reject null hypothesis. Cohen's d = 4.2/1.14 = 3.684 (very large effect).


Insights: 

- Algorithm
Step 1: Calculate means and variances
 
Step 2: Calculate standard error

Step 3: Calculate t-statistic
 
Step 4: Calculate df using Welch-Satterthwaite equation

Step 5: Calculate p-value from t-distribution

Step 6: Make decision and compute Cohen's d


"""

#SOLUTION: 

import numpy as np
from scipy.stats import t

def two_sample_t_test(sample1: list[float], sample2: list[float], 
                      alpha: float = 0.05) -> dict:
	"""
	Perform a two-sample independent t-test (Welch's t-test).
	
	Args:
		sample1: First sample data
		sample2: Second sample data
		alpha: Significance level (default 0.05)
	
	Returns:
		Dictionary containing:
		- t_statistic: The calculated t-statistic
		- p_value: Two-tailed p-value
		- degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
		- reject_null: Boolean, whether to reject null hypothesis
		- cohens_d: Effect size (Cohen's d)
	"""
    x1 = np.array(sample1, dtype=float)
    x2 = np.array(sample2, dtype=float)

    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return {}

    # Mean values of both sample
    mean1 = x1.mean()
    mean2 = x2.mean()

    #Sample Unbiased variances 
    var1 = x1.var(ddof=1)
    var2 = x2.var(ddof=1)

    # Standard error
    se = np.sqrt(var1 / n1 + var2 / n2)

    # t-statistic
    t_stat = (mean1 - mean2) / se

    # Welch–Satterthwaite degrees of freedom
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 ** 2) / (n1 ** 2 * (n1 - 1)) +
        (var2 ** 2) / (n2 ** 2 * (n2 - 1))
    )

    # Two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    # Hypothesis decision
    reject_null = p_value < alpha

    # Cohen's d (pooled standard deviation)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens = (mean1 - mean2) / pooled

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": round(df, 4),
        "reject_null": reject_null,
        "cohens_d": round(cohens, 4)
    }

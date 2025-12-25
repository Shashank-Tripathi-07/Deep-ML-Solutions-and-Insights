"""

Write a Python function that simulates the Central Limit Theorem (CLT). 
The function should take as input the number of samples, the sample size, and the distribution type ('uniform' or 'exponential').
It should return the mean of the sample means.

Example:
Input:
simulate_clt(num_samples=1000, sample_size=30, distribution='uniform')
Output:
0.4996
Reasoning:
We draw 1000 samples of size 30 each from a uniform(0,1) distribution. The mean of sample means is close to 0.5 due to the Central Limit Theorem.

Insights: 

- CLT says that, "Regardless of the original distribution (as long as it has finite mean and variance), the distribution of sample means approaches a normal distribution as the sample size increases." 
- Choose a distribution, Draw num_samples, Compute the mean of each sample and then compute the mean of those sample means. 
- Now let's code. 

""""







#SOLUTION:

#Give it a try again, you can do it...


#So you really wanna see the solution ? 


#Here you go 

import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
	"""
	Simulate the Central Limit Theorem (CLT).

	Args:
		num_samples: number of repeated samples to draw
		sample_size: size of each sample
		distribution: 'uniform' or 'exponential'

	Returns:
		Mean of the sample means (float)
	"""
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(num_samples, sample_size))
    elif distribution == 'exponential':
        samples = np.random.exponential(scale=1.0, size=(num_samples, sample_size))
    else:
        raise ValueError("distribution must be 'uniform' or 'exponential'")

    sample_means = np.mean(samples, axis=1)
    return float(np.mean(sample_means))

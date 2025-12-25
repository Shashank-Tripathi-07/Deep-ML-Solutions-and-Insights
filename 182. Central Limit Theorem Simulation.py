"""

Write a Python function to demonstrate the Central Limit Theorem (CLT). 
Your function should draw many samples from a chosen distribution, compute their sample means, standardize them to Z-scores,
and return the mean and standard deviation of these standardized values. The implementation should handle at least the following distributions: Uniform(0,1), Exponential(scale=1.0), and Bernoulli(p=0.3).

Example:
Input:
simulate_clt('exponential', n=30, runs=10000, seed=42)
Output:
{'mean': -0.003, 'std': 1.002}
Reasoning:
Drawing 10,000 samples of size 30 from an exponential distribution and

Insights: 

- You need to solve it using PyTorch. 
- I tried using Numpy but that doesn't quite work out. 
- I had to kind of trace the whole calculation in pytorch from the numpy version so as to get the question done. Nonetheless, got to learn a lot about PyTorch 

"""


#SOLUTION: 


import torch

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem using PyTorch tensors.
    """
    torch.manual_seed(seed)
    
    
    # 1. Define Parameters and Generate Samples
    if distribution == 'uniform':
        mu, sigma = 0.5, (1/12)**0.5
        # torch.rand generates Uniform(0,1)
        samples = torch.rand(runs, n)
        
    elif distribution == 'exponential':
        # PyTorch uses rate (lambda), where scale = 1/lambda. 
        # If scale=1.0, then lambda=1.0
        mu, sigma = 1.0, 1.0
        # Exponential distribution with rate=1.0
        dist = torch.distributions.Exponential(torch.tensor([1.0]))
        samples = dist.sample((runs, n)).squeeze()
        
    elif distribution == 'bernoulli':
        p = 0.3
        mu, sigma = p, (p * (1 - p))**0.5
        # torch.bernoulli takes a tensor of probabilities
        probs = torch.full((runs, n), p)
        samples = torch.bernoulli(probs)
        
    else:
        raise ValueError("Unsupported distribution")

    # 2. Calculate Sample Means (across columns)
    sample_means = torch.mean(samples, dim=1)

    # 3. Standardize to Z-scores
    # Z = (X_bar - mu) / (sigma / sqrt(n))
    z_scores = (sample_means - mu) / (sigma / (n**0.5))

    # 4. Return as standard Python floats
    return {
        'mean': float(torch.mean(z_scores)),
        'std': float(torch.std(z_scores, unbiased=False)) # Population std
    }

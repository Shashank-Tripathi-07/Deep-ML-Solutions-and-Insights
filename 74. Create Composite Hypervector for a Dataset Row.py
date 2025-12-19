"""
74. Create Composite Hypervector for a Dataset Row [Medium] 

Task: Generate a Composite Hypervector Using Hyperdimensional Computing
Your task is to implement the function create_row_hv(row, dim, random_seeds) to generate a composite hypervector for a given dataset row using Hyperdimensional Computing (HDC).
Each feature in the row is represented by binding hypervectors for the feature name and its value.
The hypervectors for the values are created using the same feature seed provided in the random_seeds dictionary to ensure reproducibility.
All feature hypervectors are then bundled to create a composite hypervector for the row.

Input:
row: A dictionary representing a dataset row, where keys are feature names and values are their corresponding values.
dim: The dimensionality of the hypervectors.
random_seeds: A dictionary where keys are feature names and values are seeds to ensure reproducibility of hypervectors.

Output:
A composite hypervector representing the entire row.

Example:
Input:
row = {"FeatureA": "value1", "FeatureB": "value2"}
dim = 5
random_seeds = {"FeatureA": 42, "FeatureB": 7}
print(create_row_hv(row, dim, random_seeds))
Output:
[ 1, -1,  1,  1,  1]

Reasoning:
The composite hypervector is created by binding hypervectors for each feature and bundling them together.


Analysis and Insights: 

- Gear up, this question is tough but I'll explain it in depth making it easier for you. 
- We've a list of features and for each feature we need to 
  - Generate a feature hypervector (for the feature's name) 
  - Generate a value hypervector for the feature value using the same seed.
  - skip one random generation to get a different vector
  - Bind them (perform an element-wise multiplication, we'll do this manually in the code) 
  - Bundle them (do an element-wise sum of feature hypervectors) 
  - Normalize positive and zero values to 1 and negative values to -1.  
- I'd suggest taking a look into the whole concept also as a part of reearch as it'll help in understand the whole scale. 

- We're basically doing an implementation of hyperdimensional symbolic encoding, embedding structured data into hyper-dimensional geometric space. 

WHERE IS IT USED ?? 
-Brain Computing/Neuromorphic Computing [Core origin of concept] 
- Classification and Pattern Recognition 
- NLP
-Graphs and Relational Data 
- Robotics and Sensor Fusion 
- memory systems and active recall 

Let's head to solution

"""
#SOLUTION: 

import numpy as np

def create_row_hv(row, dim, random_seeds):
    feature_hypervectors = []
    
    for feature_name, feature_value in row.items():
        # Get the seed for this feature
        feature_seed = random_seeds[feature_name]
        
        # Generate hypervector for the feature name (column)
        np.random.seed(feature_seed)
        column_hv = np.random.choice([-1, 1], size=dim)
        
        # Generate hypervector for the feature value using the same seed
        np.random.seed(feature_seed)
        # Skip one random generation to get different vector
        np.random.choice([-1, 1], size=dim)
        value_hv = np.random.choice([-1, 1], size=dim)
        
        # Bind: element-wise multiplication
        feature_hv = column_hv * value_hv
        
        feature_hypervectors.append(feature_hv)
    
    # Bundle: sum all feature hypervectors
    bundled_hv = np.sum(feature_hypervectors, axis=0)
    
    # Normalize: positive -> 1, negative -> -1, zero -> 1
    normalized_hv = np.where(bundled_hv > 0, 1, np.where(bundled_hv < 0, -1, 1))
    
    return normalized_hv


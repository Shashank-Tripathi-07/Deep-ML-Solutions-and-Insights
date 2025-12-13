"""
112. Min-Max Normalization of Feature Values [Easy] 

Category: Data Preprocessing 

Implement a function that performs Min-Max Normalization on a list of integers, scaling all values to the range [0, 1]. 
Min-Max normalization helps ensure that all features contribute equally to a model by scaling them to a common range.

Example:
Input:
min_max([1, 2, 3, 4, 5])
Output:
[0.0, 0.25, 0.5, 0.75, 1.0]

Reasoning:
The minimum value is 1 and the maximum is 5. Each value is scaled using the formula (x - min) / (max - min).

# I find the reasoning to be wrong as in test case 2, it requires us to use the min and max values from the input list itself, rather than hardcoding min as 1 and max as 5. Hope they correct



"""
#SOLUTION: 
def min_max(x: list[int]) -> list[float]:

    #Handling Empty case: 
    if not x:
        return []
    
    #Extracting min and max values: 
    mn=min(x)
    mx=max(x)

    #Handling ZeroDivisionError
    if mn == mx: 
        return [0.0] * len(x)
    
    #Final calculation:
    return [((v-mn)/(mx-mn)) for v in x]

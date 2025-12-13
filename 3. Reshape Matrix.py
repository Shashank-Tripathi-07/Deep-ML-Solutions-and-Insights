"""
3. Matrix Reshape [easy] 

Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list [ ]

Example:
Input:
a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
Output:
[[1, 2], [3, 4], [5, 6], [7, 8]]
Reasoning:
The given matrix is reshaped from 2x4 to 4x2.


# Insights: 

#the given input can be converted into an array and then reshaped using numpy 
# to handle the ValueError of not being able to reshape, we can use the try-return block to handle that case. It'd have been more fun if more complex and crazier test cases would've been involved. 
# I also learnt something new about the try-except use cases here. 


#SOLUTION: 

""" 

import numpy as np 

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:

    x = np.array(a)

    try: 

        reshaped_matrix = x.reshape(new_shape)
        return reshaped_matrix
      
    except ValueError: 
        return []

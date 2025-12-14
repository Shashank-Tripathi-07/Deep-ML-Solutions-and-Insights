""""

9. Matrix times matrix [medium]

multiply two matrices together (return -1 if shapes of matrix don't align), i.e. C=A⋅B -> C=A⋅B

Example:
Input:
A = [[1,2],[2,4]], B = [[2,1],[3,4]]
Output:
[[ 8,  9],[16, 18]]
Reasoning:
1*2 + 2*3 = 8; 2*2 + 3*4 = 16; 1*1 + 2*4 = 9; 2*1 + 4*4 = 18 Example 2: input: A = [[1,2], [2,4]], B = [[2,1], [3,4], [4,5]] output: -1 reasoning: the length of the rows of A does not equal the column length of B

Insights: 

# We need to change it to list at last for perfect output. 
# The problem could be enhanced with more test cases so as to make the medium question a bit more medium 
# The question is fun :) 


SOLUTION: 

"""


import numpy as np 

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	
    A = np.array(a, dtype = int)   #something new is to define the dtype to ease the calculations further
    B = np.array(b, dtype = int)
    
    try:
        c = (A @ B)
        return c.tolist()
    
    except ValueError: 
        return -1 

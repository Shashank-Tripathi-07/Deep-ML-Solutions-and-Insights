"""
57. Gauss-Seidel Method for Solving Linear Systems [Medium]

Task: Implement the Gauss-Seidel Method
Your task is to implement the Gauss-Seidel method, an iterative technique for solving a system of linear equations (Ax = b).
The function should iteratively update the solution vector (x) by using the most recent values available during the iteration process.
Write a function gauss_seidel(A, b, n, x_ini=None) where:

A is a square matrix of coefficients,
b is the right-hand side vector,
n is the number of iterations,
x_ini is an optional initial guess for (x) (if not provided, assume a vector of zeros).
The function should return the approximated solution vector (x) after performing the specified number of iterations.

Example:
Input:
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)

n = 100
print(gauss_seidel(A, b, n))
Output:
# [0.2, 1.4, 0.8]  (Approximate, values may vary depending on iterations)

Reasoning:
The Gauss-Seidel method iteratively updates the solution vector (x) until convergence. The output is an approximate solution to the linear system.



Analysis, Learning and Insights: 

- we are assuming:
  -A is square
  -A and b are dimensionally compatible
  -The system has exactly one unknown per equation

These assumptions are standard for Gauss–Seidel and are usually stated upfront in numerical methods.

Gauss–Seidel is derived from the decomposition

𝐴=𝐿+𝐷+𝑈
where:

D = diagonal of 𝐴
L = strictly lower-triangular part
U = strictly upper-triangular part

The iteration formula is:

(𝐷+𝐿)𝑥^((𝑘+1))=𝑏−𝑈𝑥(𝑘)(D+L)x(k+1)=b−Ux^(k)

or equivalently:

𝑥^(𝑘+1)=(𝐷+𝐿)^(−1)(𝑏−𝑈𝑥^(𝑘))

The code never computes inverses; it performs this update equation component-wise.

""" 


#Solution:

import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	
     y = b.shape[0]
 
     # --- Initial guess ---
     if x_ini is None:
         x = np.zeros(y, dtype=float)
     else:
         x = np.asarray(x_ini, dtype=float)
         if x.shape[0] != y:
             raise ValueError("Initial guess x_ini has incompatible size")

     # --- Iterative solve ---
     for _ in range(n):
         x_old = x.copy()

         for i in range(y):
             s1 = np.dot(A[i, :i], x[:i])       # updated values (mathematically the lower triangular contribution)
             s2 = np.dot(A[i, i+1:], x_old[i+1:])  # previous iteration values (mathematically the upper triangular contribution) 

             x[i] = (b[i] - s1 - s2) / A[i, i]    #update equation 

     return x.tolist()

# You'll get to learn a lot in this question. 

"""

28. SVD of a 2x2 Matrix using eigen values & vectors [Hard]

Given a 2x2 matrix, write a Python function to compute its Singular Value Decomposition (SVD). 
The function should return the matrices U, S, and V such that A = U * S * V, use the method described in this post 
https://metamerist.blogspot.com/2006/10/linear-algebra-for-graphics-geeks-svd.html

Example:
Input:
A = [[-10, 8], 
         [10, -1]]
Output:
(array([[  0.8, -0.6], [-0.6, -0.8]]), 
    array([15.65247584,  4.47213595]), 
    array([[ -0.89442719,  0.4472136], [ -0.4472136 , -0.89442719]]))
    
Reasoning:
The SVD of the matrix A is calculated using the eigenvalues and eigenvectors of A^T A and A A^T. 
The singular values are the square roots of the eigenvalues, and the eigenvectors form the columns of matrices U and V.


Insights: 

- I delved into the maths again, and found the calculation at last. tried a lot and got it to work.


Solution: 

"""


import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
	
	A = np.array(A, dtype=float)
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    # Step 1: Compute U from A * A^T
    # theta = 0.5 * atan2(2ac + 2bd, a^2 + b^2 - c^2 - d^2)
    theta = 0.5 * np.arctan2(2*a*c + 2*b*d, a**2 + b**2 - c**2 - d**2)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    U = np.array([[c_theta, -s_theta],
                  [s_theta, c_theta]])
    
    # Step 2: Compute singular values
    # S1 = a^2 + b^2 + c^2 + d^2
    # S2 = sqrt((a^2 + b^2 - c^2 - d^2)^2 + 4(ac + bd)^2)
    S1 = a**2 + b**2 + c**2 + d**2
    S2 = np.sqrt((a**2 + b**2 - c**2 - d**2)**2 + 4*(a*c + b*d)**2)
    
    # sigma1 = sqrt((S1 + S2) / 2)
    # sigma2 = sqrt((S1 - S2) / 2)
    sigma1 = np.sqrt((S1 + S2) / 2)
    sigma2 = np.sqrt((S1 - S2) / 2)
    S = np.array([sigma1, sigma2])
    
    # Step 3: Compute V
    # First compute temporary angle Phi
    # Phi = 0.5 * atan2(2ab + 2cd, a^2 - b^2 + c^2 - d^2)
    Phi = 0.5 * np.arctan2(2*a*b + 2*c*d, a**2 - b**2 + c**2 - d**2)
    c_Phi = np.cos(Phi)
    s_Phi = np.sin(Phi)
    
    # Compute s11 and s22 to determine signs
    # s11 = (a*cos(theta) + c*sin(theta))*cos(Phi) + (b*cos(theta) + d*sin(theta))*sin(Phi)
    # s22 = (a*sin(theta) - c*cos(theta))*sin(Phi) + (-b*sin(theta) + d*cos(theta))*cos(Phi)
    s11 = (a*c_theta + c*s_theta)*c_Phi + (b*c_theta + d*s_theta)*s_Phi
    s22 = (a*s_theta - c*c_theta)*s_Phi + (-b*s_theta + d*c_theta)*c_Phi
    
    # Compute V with sign corrections
    # sign function: returns 0 if value is 0, otherwise returns sign
    sign_s11 = np.sign(s11) if s11 != 0 else 0
    sign_s22 = np.sign(s22) if s22 != 0 else 0
    
    V = np.array([[sign_s11 * c_Phi, -sign_s22 * s_Phi],
                  [sign_s11 * s_Phi,  sign_s22 * c_Phi]])
    
  
    return (U,S,V)

===================================================================================================================




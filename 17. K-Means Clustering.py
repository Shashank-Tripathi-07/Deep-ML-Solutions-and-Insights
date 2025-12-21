"""
17. K-Means Clustering [Medium] 

Your task is to write a Python function that implements the k-Means clustering algorithm.
This function should take specific inputs and produce a list of final centroids.
k-Means clustering is a method used to partition n points into k clusters.
The goal is to group similar points together and represent each group by its center (called the centroid).

Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.

Example:
Input:
points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
Output:
[(1, 2), (10, 2)]
Reasoning:
Given the initial centroids and a maximum of 10 iterations,
the points are clustered around these points,
and the centroids are updated to the mean of the assigned points,
resulting in the final centroids which approximate the means of the two clusters.
The exact number of iterations needed may vary, but the process will stop after 10 iterations at most.



Insights: 

- I may be saying this for the 1000th time by now, but read the alogorithm, how it works and why it works in that manner and you'll arrive to the solution. 


"""
#SOLUTION: 

import numpy as np 

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	
    point = np.array(points, dtype=float)
    initial_centroid = np.array(initial_centroids, dtype=float)
    n_points, dim = point.shape
    
    for _ in range(max_iterations): 
        clusters = [[] for _ in range(k)]

        for point in points: 
            #compute euclidean distance squared
            distances = np.sum((initial_centroid - point)**2,axis=1)
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        #update step
        new_centroids = initial_centroid.copy()

        for i in range(k):
            if clusters[i]:
                new_centroids[i]=np.mean(clusters[i],axis=0)
  
        final_centroids = new_centroids 

	return [(np.round(final_centroids, 4)).tolist()]


import cupy as cp
import numpy as np

from mls_1.cupy.kmeans import *
from mls_1.utils import *
from distances import np_distance_function_ref

seed = 1337
np.random.seed(seed)
cp.random.seed(seed)


def kmeans_np(datapoints, k=10, distance_metric='l2', max_iter=100, tol=1e-6, initialized_centroids=None):

    n_samples, n_features = datapoints.shape
    distance_function = np_distance_function_ref[distance_metric]
    
    # Randomly initialize centroids
    if initialized_centroids is not None:
        centroids = initialized_centroids
    else:
        centroids = datapoints[np.random.choice(n_samples, k, replace=False)]
    
    for i in range(max_iter):
        new_centroids = np.zeros((k, n_features))
        distances = np.zeros((n_samples, k))
        for j in range(k):
            cluster_distances = distance_function(datapoints, centroids[j][None,:])
            distances[:, j] = cluster_distances.flatten()
        # Assign each datapoint to the nearest centroid
        labels = np.argmin(distances, axis=1)
        # Compute new centroids
        new_centroids = np.array([datapoints[labels == j].mean(axis=0) for j in range(k)])
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol: 
            print("Converged after", i, "iterations")
            break
        centroids = new_centroids
    
    return centroids, labels

def kmeans_np_vec(datapoints, k=10, distance_metric='l2', max_iter=100, tol=1e-6, initialized_centroids=None):

    n_samples, n_features = datapoints.shape
    distance_function = np_distance_function_ref[distance_metric]

    if initialized_centroids is not None:
        centroids = initialized_centroids
    else:
        centroids = datapoints[np.random.choice(n_samples, k, replace=False)]
    
    for i in range(max_iter):
        distances = distance_function(datapoints[:, None, :], centroids[None, :, :]).squeeze()
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([datapoints[labels==j].mean(axis=0) for j in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol: 
            print("Converged after", i, "iterations")
            break
        centroids = new_centroids
    
    return centroids, labels

def test_kmeans(datapoints, k, distance_metric, tol):

    initialized_centroids = datapoints[np.random.choice(datapoints.shape[0], k, replace=False)] # ensure these are intialised to the same point

    (numpy_centroids, numpy_labels), time = time_function(kmeans_np, 
        datapoints, k=k, distance_metric=distance_metric, tol=tol, initialized_centroids=initialized_centroids)
    (numpy_centroids_vec, numpy_labels_vec), time = time_function(kmeans_np_vec, 
        datapoints, k=k, distance_metric=distance_metric, tol=tol, initialized_centroids=initialized_centroids)
    (cp_centroids, cp_labels), time = time_function(kmeans, 
        cp.array(datapoints), k=k, distance_metric=distance_metric, tol=tol, initialized_centroids=cp.array(initialized_centroids))

    numpy_centroids_sorted = np.sort(numpy_centroids, axis=0)
    numpy_centroids_vec_sorted = np.sort(numpy_centroids_vec, axis=0)
    cp_centroids_sorted = np.sort(cp.asnumpy(cp_centroids), axis=0)
    assert np.allclose(numpy_centroids_sorted, numpy_centroids_vec_sorted), f"Centroids mismatch: {numpy_centroids_sorted} vs {numpy_centroids_vec_sorted}"
    assert np.allclose(numpy_centroids_sorted, cp_centroids_sorted), f"Centroids mismatch: {numpy_centroids_sorted} vs {cp_centroids_sorted}"
    
    return numpy_centroids, numpy_labels

def test():
    
    n_dimensions = int(1e4)
    n_points = 1000
    variance = 10

    datapoints = np.random.rand(n_points, n_dimensions) * variance

    cluster_centroids, labels = test_kmeans(datapoints, k=10, distance_metric='cosine', tol=1e-9)
    cluster_centroids, labels = test_kmeans(datapoints, k=10, distance_metric='l2', tol=1e-9)

if __name__ == "__main__":
    test()
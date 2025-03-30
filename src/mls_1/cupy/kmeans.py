import cupy as cp 

from mls_1.cupy.knn import distance_function_ref

def kmeans(datapoints, k=10, distance_metric='l2', max_iter=100, tol=1e-9, initialized_centroids=None):

    n_samples, n_features = datapoints.shape
    distance_function = distance_function_ref[distance_metric]
    
    if initialized_centroids is not None:
        centroids = initialized_centroids
    else:
        centroids = datapoints[cp.random.choice(n_samples, k, replace=False)]
    
    for i in range(max_iter):
        
        # allocate clusters
        distances = distance_function(datapoints[:, None, :], centroids[None, :, :]).squeeze()
        labels = cp.argmin(distances, axis=1)

        # update centroids
        new_centroids = cp.array([datapoints[labels == j].mean(axis=0) for j in range(k)])

        # check for convergence
        if cp.linalg.norm(new_centroids - centroids) < tol: 
            print("Converged after", i, "iterations")
            break

        centroids = new_centroids
    
    return centroids, labels

import torch

from mls_1.torch.knn import distance_function_ref

def kmeans(datapoints, k=10, distance_metric='l2', max_iter=100, tol=1e-9, initialized_centroids=None):

    n_samples, n_features = datapoints.shape
    distance_function = distance_function_ref[distance_metric]
    
    if initialized_centroids is not None:
        centroids = initialized_centroids
    else:
        centroids = datapoints[torch.randperm(n_samples)[:k]]
    
    for i in range(max_iter):
        
        # allocate clusters
        distances = distance_function(datapoints[:, None, :], centroids[None, :, :]).squeeze()
        labels = torch.argmin(distances, axis=1)
        # update centroids
        new_centroids = torch.stack([datapoints[labels == j].mean(axis=0) for j in range(k)])

        # check for convergence
        if torch.linalg.norm(new_centroids - centroids) < tol: 
            print("Converged after", i, "iterations")
            break

        centroids = new_centroids
    
    return centroids, labels

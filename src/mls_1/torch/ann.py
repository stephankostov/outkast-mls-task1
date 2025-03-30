import torch

from mls_1.torch.distances import distance_function_ref
from mls_1.torch.kmeans import kmeans
from mls_1.torch.knn import knn

def ann(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric):

    original_idxs = torch.arange(datapoints.shape[0])

    target_cluster_idxs = original_idxs[labels == target_label]
    k1_idxs = target_cluster_idxs[knn(datapoints[target_cluster_idxs], target_vector, k, distance_metric)]

    # exclude the target cluster centroid
    non_target_cluster_centroid_idxs = torch.tensor([cluster_centroid_idx for cluster_centroid_idx in range(cluster_centroids.shape[0]) if cluster_centroid_idx != target_label])
    closest_cluster_centroid_idxs = non_target_cluster_centroid_idxs[knn(cluster_centroids[non_target_cluster_centroid_idxs], target_vector, k=1, distance_metric=distance_metric)]
    
    closest_cluster_idxs = original_idxs[labels == closest_cluster_centroid_idxs[0]]
    k2_idxs = closest_cluster_idxs[knn(datapoints[closest_cluster_idxs], target_vector, k, distance_metric)]

    k1k2_idxs = torch.sort(torch.concatenate((k1_idxs, k2_idxs)))[0]
    k1k2_nearest_neighbour_idxs = k1k2_idxs[knn(datapoints[k1k2_idxs], target_vector, k, distance_metric)]

    return k1k2_nearest_neighbour_idxs

def ann_full(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
    """
    A[N, D]: A collection of vectors
    X: A specified vector
    K: Top K
    """
    cluster_centroids, labels = kmeans(datapoints, k, distance_metric=distance_metric, max_iter=max_iter, tol=tol)
    target_label = torch.argmin(distance_function_ref[distance_metric](target_vector, cluster_centroids))
    ann_idxs = ann(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric)
    return ann_idxs

def ann_stream(datapoints,  labels, cluster_centroids, target_vector, target_label, k, distance_metric='l2'):
    """
    A[N, D]: A collection of vectors
    X: A specified vector
    K: Top K
    """

    original_idxs = torch.arange(datapoints.shape[0])

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    with stream1:
        target_cluster_idxs = original_idxs[labels == target_label]
        k1_idxs = target_cluster_idxs[knn(datapoints[target_cluster_idxs], target_vector, k, distance_metric)]

    with stream2:
        # # exclude the target cluster centroid
        # non_target_cluster_centroid_idxs = cp.array([cluster_centroid_idx for cluster_centroid_idx in range(cluster_centroids.shape[0]) if cluster_centroid_idx != target_label])
        closest_cluster_centroid_idxs = knn(cluster_centroids, target_vector, k=2, distance_metric=distance_metric)
        closest_cluster_centroid_idxs = closest_cluster_centroid_idxs[closest_cluster_centroid_idxs!=target_label]  # Exclude the target cluster 
        
        closest_cluster_idxs = original_idxs[labels == closest_cluster_centroid_idxs[0]]
        k2_idxs = closest_cluster_idxs[knn(datapoints[closest_cluster_idxs], target_vector, k, distance_metric)]

    torch.cuda.Stream(null=True).synchronize()
    k1k2_idxs = torch.sort(torch.concatenate((k1_idxs, k2_idxs)))[0]
    k1k2_nearest_neighbour_idxs = k1k2_idxs[knn(datapoints[k1k2_idxs], target_vector, k, distance_metric)]

    return k1k2_nearest_neighbour_idxs

def ann_full_stream(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
    """
    A[N, D]: A collection of vectors
    X: A specified vector
    K: Top K
    """
    cluster_centroids, labels = kmeans(datapoints, k, distance_metric=distance_metric, max_iter=max_iter, tol=tol)
    target_label = torch.argmin(distance_function_ref[distance_metric](target_vector, cluster_centroids))
    ann_idxs = ann(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric)
    return ann_idxs
# def ann_full_stream(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
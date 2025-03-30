import cupy as cp

from mls_1.cupy.distances import distance_function_ref
from mls_1.cupy.kmeans import kmeans
from mls_1.cupy.knn import knn

def ann(datapoints, labels, cluster_centroids, target_vector, k, distance_metric):

    original_idxs = cp.arange(datapoints.shape[0])

    nearest_clusters = knn(cluster_centroids, target_vector, k=2, distance_metric=distance_metric)

    cluster_knn_idxs = cp.array([], dtype=cp.int32)
    for c in nearest_clusters:
        target_cluster_idxs = original_idxs[labels == c]
        idxs = target_cluster_idxs[knn(datapoints[target_cluster_idxs], target_vector, k, distance_metric)].squeeze()
        cluster_knn_idxs = cp.concatenate((cluster_knn_idxs, idxs))

    ann_idxs = cluster_knn_idxs[knn(datapoints[cluster_knn_idxs], target_vector, k, distance_metric)]
    return ann_idxs

def ann_full(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
    """
    A[N, D]: A collection of vectors
    X: A specified vector
    K: Top K
    """
    cluster_centroids, labels = kmeans(datapoints, k, distance_metric=distance_metric, max_iter=max_iter, tol=tol)
    ann_idxs = ann(datapoints, labels, cluster_centroids, target_vector, k, distance_metric)
    return ann_idxs

def ann_stream(datapoints,  labels, cluster_centroids, target_vector, k, distance_metric='l2'):
    """
    A[N, D]: A collection of vectors
    X: A specified vector
    K: Top K
    """

    original_idxs = cp.arange(datapoints.shape[0])

    nearest_clusters = knn(cluster_centroids, target_vector, k=2, distance_metric=distance_metric)

    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()

    with stream1:
        target_cluster_idxs = original_idxs[labels == nearest_clusters[0]]
        k1_idxs = target_cluster_idxs[knn(datapoints[target_cluster_idxs], target_vector, k=k, distance_metric=distance_metric)].squeeze()

    with stream2:
        closest_cluster_idxs = original_idxs[labels == nearest_clusters[1]]
        k2_idxs = closest_cluster_idxs[knn(datapoints[closest_cluster_idxs], target_vector, k=k, distance_metric=distance_metric)].squeeze()

    cp.cuda.Stream(null=True).synchronize()
    k1k2_idxs = cp.concatenate((k1_idxs, k2_idxs))
    ann_idxs = k1k2_idxs[knn(datapoints[k1k2_idxs], target_vector, k=k, distance_metric=distance_metric)]

    return ann_idxs

def ann_full_stream(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
    """
    A[N, D]: A collection of vectors
    X: A specified vector
    K: Top K
    """
    cluster_centroids, labels = kmeans(datapoints, k, distance_metric=distance_metric, max_iter=max_iter, tol=tol)
    target_label = cp.argmin(distance_function_ref[distance_metric](target_vector, cluster_centroids))
    ann_idxs = ann(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric)
    return ann_idxs
# def ann_full_stream(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
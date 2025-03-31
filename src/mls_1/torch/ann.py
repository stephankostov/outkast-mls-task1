import torch

from mls_1.torch.distances import distance_function_ref
from mls_1.torch.kmeans import kmeans
from mls_1.torch.knn import knn

def ann(datapoints, labels, cluster_centroids, target_vector, k, distance_metric):

    original_idxs = torch.arange(datapoints.shape[0], device=datapoints.device)

    nearest_clusters = knn(cluster_centroids, target_vector, k=2, distance_metric=distance_metric)

    cluster_knn_idxs = torch.tensor([], dtype=torch.int32, device=datapoints.device)
    for c in nearest_clusters:
        target_cluster_idxs = original_idxs[labels == c]
        idxs = target_cluster_idxs[knn(datapoints[target_cluster_idxs], target_vector, k, distance_metric)].squeeze()
        cluster_knn_idxs = torch.cat((cluster_knn_idxs, idxs.to(torch.int32)))

    ann_idxs = cluster_knn_idxs[knn(datapoints[cluster_knn_idxs], target_vector, k, distance_metric)]
    return ann_idxs

def ann_full(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):
    cluster_centroids, labels = kmeans(datapoints, k, distance_metric=distance_metric, max_iter=max_iter, tol=tol)
    ann_idxs = ann(datapoints, labels, cluster_centroids, target_vector, k, distance_metric)
    return ann_idxs

def ann_stream(datapoints,  labels, cluster_centroids, target_vector, k, distance_metric='l2'):

    original_idxs = torch.arange(datapoints.shape[0], device=datapoints.device)

    nearest_clusters = knn(cluster_centroids, target_vector, k=2, distance_metric=distance_metric)

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    with torch.cuda.stream(stream1):
        target_cluster_idxs = original_idxs[labels == nearest_clusters[0]]
        k1_idxs = target_cluster_idxs[knn(datapoints[target_cluster_idxs], target_vector, k=k, distance_metric=distance_metric)].squeeze()

    with torch.cuda.stream(stream2):
        closest_cluster_idxs = original_idxs[labels == nearest_clusters[1]]
        k2_idxs = closest_cluster_idxs[knn(datapoints[closest_cluster_idxs], target_vector, k=k, distance_metric=distance_metric)].squeeze()

    torch.cuda.synchronize()
    k1k2_idxs = torch.cat((k1_idxs.to(torch.int32), k2_idxs.to(torch.int32)))
    ann_idxs = k1k2_idxs[knn(datapoints[k1k2_idxs], target_vector, k=k, distance_metric=distance_metric)]

    return ann_idxs

def ann_full_stream(datapoints, target_vector, k, distance_metric='l2', max_iter=100, tol=1e-9):

    cluster_centroids, labels = kmeans(datapoints, k, distance_metric=distance_metric, max_iter=max_iter, tol=tol)
    ann_idxs = ann_stream(datapoints, labels, cluster_centroids, target_vector, k, distance_metric)
    return ann_idxs

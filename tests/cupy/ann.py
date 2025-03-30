import cupy as cp
import numpy as np

from mls_1.utils import *  # Replace '*' with the specific function(s) you want to import
from mls_1.cupy.ann import *
from distances import np_distance_function_ref
from knn import knn_np

seed = 1337
np.random.seed(seed)
cp.random.seed(seed)

def ann_np(datapoints, labels, cluster_centroids, target_vector, k, distance_metric):

    original_idxs = np.arange(datapoints.shape[0])

    nearest_cluster_centroids = knn_np(cluster_centroids, target_vector, k=2, distance_metric=distance_metric)

    knn_idxs = np.array([], dtype=np.int32)
    for c in nearest_cluster_centroids:
        target_cluster_idxs = original_idxs[labels == c]
        idxs = target_cluster_idxs[knn_np(datapoints[target_cluster_idxs], target_vector, k, distance_metric)].squeeze()
        knn_idxs = np.concatenate((knn_idxs, idxs))

    ann_idxs = knn_idxs[knn_np(datapoints[knn_idxs], target_vector, k, distance_metric)]
    return ann_idxs
    
def test_ann(datapoints, labels, cluster_centroids, target_vector, k, distance_metric):

    nearest_datapoint_idxs_np, time = time_function(ann_np, datapoints, labels, cluster_centroids, target_vector, k, distance_metric)
    nearest_datapoint_idxs_cp, time = time_function(ann, cp.array(datapoints), cp.array(labels), cp.array(cluster_centroids), cp.array(target_vector), k, distance_metric)

    assert np.allclose(nearest_datapoint_idxs_np, nearest_datapoint_idxs_cp), f"Centroids mismatch: {nearest_datapoint_idxs_np} vs {nearest_datapoint_idxs_cp}"

def test_ann_stream(datapoints, labels, cluster_centroids, target_vector, k, distance_metric):

    datapoints = cp.array(datapoints); labels = cp.array(labels); cluster_centroids = cp.array(cluster_centroids); target_vector = cp.array(target_vector)

    nearest_datapoint_idxs_cp, time = time_function(ann, 
        datapoints, labels, cluster_centroids, target_vector, k, distance_metric)
    nearest_datapoint_idxs_cp_stream, time = time_function(ann_stream, 
        datapoints, labels, cluster_centroids, target_vector, k, distance_metric)

    assert np.allclose(nearest_datapoint_idxs_cp, nearest_datapoint_idxs_cp_stream), f"Centroids mismatch: {nearest_datapoint_idxs_cp} vs {nearest_datapoint_idxs_cp_stream}"

def test_ann_full(datapoints, target_vector, k, distance_metric, **kwargs):

    datapoints = cp.array(datapoints)
    target_vector = cp.array(target_vector)

    ann_idxs, time = time_function(ann_full, datapoints, target_vector, k, distance_metric=distance_metric, **kwargs)
    knn_idxs, time = time_function(knn, datapoints, target_vector, k, distance_metric=distance_metric)

    ann_idxs = ann_idxs.get()
    knn_idxs = knn_idxs.get()

    recall_rate = np.sum(np.isin(ann_idxs, knn_idxs)) / ann_idxs.shape[0]
    print(f"Recall rate: {recall_rate}")
    assert recall_rate >= 0.7, f"Recall rate is too low: {recall_rate}, {knn_idxs} vs {ann_idxs}"

def test():
    
    n_dimensions = int(1e4)
    n_points = 1000
    variance = 10
    k = 6
    
    datapoints = np.random.rand(n_points, n_dimensions) * variance
    target_vector=datapoints[0][None,:]

    cluster_centroids, labels = kmeans(cp.array(datapoints), k=k, distance_metric='cosine')
    cluster_centroids = cp.asnumpy(cluster_centroids); labels = cp.asnumpy(labels)

    test_ann(datapoints, labels, cluster_centroids, k=k,  target_vector=target_vector, distance_metric='cosine')
    test_ann_full(datapoints, target_vector=target_vector, k=k, distance_metric='cosine', max_iter=100, tol=1e-9)
    test_ann_stream(datapoints, labels, cluster_centroids, target_vector=target_vector, k=k, distance_metric='cosine')

if __name__ == "__main__":
    test()
import cupy as cp
import numpy as np

from mls_1.utils import *  # Replace '*' with the specific function(s) you want to import
from mls_1.cupy.ann import *
from distances import np_distance_function_ref
from knn import knn_np

seed = 1337
np.random.seed(seed)
cp.random.seed(seed)

def ann_np(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric):

    original_idxs = np.arange(datapoints.shape[0])

    target_cluster_idxs = original_idxs[labels == target_label]
    k1_idxs = target_cluster_idxs[knn_np(datapoints[target_cluster_idxs], target_vector, k, distance_metric)]

    # exclude the target cluster centroid
    non_target_cluster_centroid_idxs = np.array([cluster_centroid_idx for cluster_centroid_idx in range(cluster_centroids.shape[0]) if cluster_centroid_idx != target_label])
    closest_cluster_centroid_idxs = non_target_cluster_centroid_idxs[knn_np(cluster_centroids[non_target_cluster_centroid_idxs], target_vector, k=1, distance_metric=distance_metric)]
    
    closest_cluster_idxs = original_idxs[labels == closest_cluster_centroid_idxs[0]]
    k2_idxs = closest_cluster_idxs[knn_np(datapoints[closest_cluster_idxs], target_vector, k, distance_metric)]

    k1k2_idxs = np.sort(np.concatenate((k1_idxs, k2_idxs)))
    k1k2_nearest_neighbour_idxs = k1k2_idxs[knn_np(datapoints[k1k2_idxs], target_vector, k, distance_metric)]
    return k1k2_nearest_neighbour_idxs
    
def test_ann(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric):

    nearest_datapoint_idxs_np, time = time_function(ann_np, datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric)
    nearest_datapoint_idxs_cp, time = time_function(ann, cp.array(datapoints), cp.array(labels), cp.array(cluster_centroids), cp.array(target_vector), cp.array(target_label), k, distance_metric)

    assert np.allclose(nearest_datapoint_idxs_np, nearest_datapoint_idxs_cp), f"Centroids mismatch: {nearest_datapoint_idxs_np} vs {nearest_datapoint_idxs_cp}"

def test_ann_stream(datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric):

    datapoints = cp.array(datapoints); labels = cp.array(labels); cluster_centroids = cp.array(cluster_centroids); target_vector = cp.array(target_vector); target_label = cp.array(target_label)

    nearest_datapoint_idxs_np, time = time_function(ann, 
        datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric)
    nearest_datapoint_idxs_cp_stream, time = time_function(ann_stream, 
        datapoints, labels, cluster_centroids, target_vector, target_label, k, distance_metric)

    assert np.allclose(nearest_datapoint_idxs_np, nearest_datapoint_idxs_cp_stream), f"Centroids mismatch: {nearest_datapoint_idxs_np} vs {nearest_datapoint_idxs_cp_stream}"

def test_ann_full(datapoints, target_vector, k, distance_metric, **kwargs):

    datapoints = cp.array(datapoints)
    target_vector = cp.array(target_vector)

    ann_idxs, time = time_function(ann_full, datapoints, target_vector, k, **kwargs)
    knn_idxs, time = time_function(knn, datapoints, target_vector, k, distance_metric=distance_metric)

    ann_idxs = ann_idxs.get()
    knn_idxs = knn_idxs.get()

    recall_rate = np.sum(np.isin(knn_idxs, ann_idxs)) / knn_idxs.shape[0]
    print(f"Recall rate: {recall_rate}")
    assert recall_rate >= 0.7, f"Recall rate is too low: {recall_rate}, {knn_idxs} vs {ann_idxs}"

def test():
    
    n_dimensions = int(1e4)
    n_points = 1000
    variance = 10
    
    datapoints = np.random.rand(n_points, n_dimensions) * variance
    target_vector=datapoints[0][None,:]

    cluster_centroids, labels = kmeans(cp.array(datapoints), k=10, distance_metric='l2')
    cluster_centroids = cp.asnumpy(cluster_centroids); labels = cp.asnumpy(labels)

    test_ann(datapoints, labels, cluster_centroids, k=10,  target_vector=target_vector, target_label=labels[0], distance_metric='l2')
    test_ann_full(datapoints, target_vector=target_vector, k=10, distance_metric='l2', max_iter=100, tol=1e-9)
    test_ann_stream(datapoints, labels, cluster_centroids, target_vector=target_vector, target_label=labels[0], k=10, distance_metric='l2')

if __name__ == "__main__":
    test()
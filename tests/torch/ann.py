import torch
import numpy as np

from mls_1.utils import *  # Replace '*' with the specific function(s) you want to import
from mls_1.torch.ann import *
from distances import np_distance_function_ref
from knn import knn_np

seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda': print("WARNING: CUDA is not available. Running on CPU.")

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
    datapoints_torch = torch.tensor(datapoints, device=device)
    labels_torch = torch.tensor(labels, device=device)
    cluster_centroids_torch = torch.tensor(cluster_centroids, device=device)
    target_vector_torch = torch.tensor(target_vector, device=device)
    nearest_datapoint_idxs_torch, time = time_function(ann, 
        datapoints_torch, labels_torch, cluster_centroids_torch, target_vector_torch, k, distance_metric)
    nearest_datapoint_idxs_torch = nearest_datapoint_idxs_torch.cpu().numpy()

    assert np.allclose(nearest_datapoint_idxs_np, nearest_datapoint_idxs_torch), f"Centroids mismatch: {nearest_datapoint_idxs_np} vs {nearest_datapoint_idxs_torch}"

def test_ann_stream(datapoints, labels, cluster_centroids, target_vector, k, distance_metric):

    datapoints_torch = torch.tensor(datapoints, device=device)
    labels_torch = torch.tensor(labels, device=device)
    cluster_centroids_torch = torch.tensor(cluster_centroids, device=device)
    target_vector_torch = torch.tensor(target_vector, device=device)

    nearest_datapoint_idxs_torch, time = time_function(ann, 
        datapoints_torch, labels_torch, cluster_centroids_torch, target_vector_torch, k, distance_metric)
    nearest_datapoint_idxs_torch_stream, time = time_function(ann_stream, 
        datapoints_torch, labels_torch, cluster_centroids_torch, target_vector_torch, k, distance_metric)

    assert np.allclose(nearest_datapoint_idxs_torch.cpu().numpy(), nearest_datapoint_idxs_torch_stream.cpu().numpy()), f"Centroids mismatch: {nearest_datapoint_idxs_torch} vs {nearest_datapoint_idxs_torch_stream}"

def test_ann_threshold(datapoints, target_vector, k, distance_metric, **kwargs):

    datapoints_torch = torch.tensor(datapoints, device=device)
    target_vector_torch = torch.tensor(target_vector, device=device)

    ann_idxs, time = time_function(ann_full, datapoints_torch, target_vector_torch, k, distance_metric=distance_metric, **kwargs)
    knn_idxs, time = time_function(knn, datapoints_torch, target_vector_torch, k, distance_metric=distance_metric)

    ann_idxs = ann_idxs.cpu().numpy()
    knn_idxs = knn_idxs.cpu().numpy()

    recall_rate = np.sum(np.isin(ann_idxs, knn_idxs)) / ann_idxs.shape[0]
    print(f"Recall rate: {recall_rate}; {knn_idxs.squeeze()} vs {ann_idxs.squeeze()}")
    if recall_rate < 0.7: print(f"WARNING: Recall rate is below threshold.")

def test_ann_full_stream(datapoints, target_vector, k, distance_metric, **kwargs):

    datapoints_torch = torch.tensor(datapoints, device=device)
    target_vector_torch = torch.tensor(target_vector, device=device)

    ann_idxs, time = time_function(ann_full, datapoints_torch, target_vector_torch, k, distance_metric=distance_metric, **kwargs)
    knn_idxs, time = time_function(ann_full_stream, datapoints_torch, target_vector_torch, k, distance_metric=distance_metric)

    # testing time saved

def test():
    
    n_dimensions = int(1e4)
    n_points = 10000
    variance = 10
    k = 7
    
    datapoints = np.random.rand(n_points, n_dimensions) * variance
    target_vector=datapoints[0][None,:]

    cluster_centroids, labels = kmeans(torch.tensor(datapoints).cuda(), k=k, distance_metric='cosine')
    cluster_centroids = cluster_centroids.cpu().numpy(); labels = labels.cpu().numpy()

    test_ann(datapoints, labels, cluster_centroids, k=k,  target_vector=target_vector, distance_metric='cosine')
    test_ann_threshold(datapoints, target_vector=target_vector, k=k, distance_metric='cosine', max_iter=100, tol=1e-9)
    test_ann_stream(datapoints, labels, cluster_centroids, target_vector=target_vector, k=k, distance_metric='cosine')
    test_ann_full_stream(datapoints, target_vector=target_vector, k=k, distance_metric='cosine', max_iter=100, tol=1e-9)

if __name__ == "__main__":
    test()
import torch
import numpy as np

from mls_1.utils import *  # Replace '*' with the specific function(s) you want to import
from mls_1.torch.knn import *
from distances import np_distance_function_ref

seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

def knn_np(datapoints, target, k, distance_metric):
    distance_function = np_distance_function_ref[distance_metric]
    distances = distance_function(datapoints, target)
    nearest_neighbours = np.argsort(distances, axis=0)[:k]
    return nearest_neighbours
    
def test_knn(datapoints, target, k, distance_metric):

    np_idxs, t = time_function(knn_np, datapoints, target, k, distance_metric)
    cp_idxs, t = time_function(knn, torch.tensor(datapoints), torch.tensor(target), k, distance_metric)
    
    assert np.allclose(np_idxs, cp_idxs), f"Indices mismatch: {np_idxs} vs {cp_idxs}"

def test():
    
    n_dimensions = int(1e4)
    n_points = 1000
    variance = 10

    datapoints = np.random.rand(n_points, n_dimensions) * variance
    test_knn(datapoints, target=datapoints[0][None,:], k=5, distance_metric='cosine')
    test_knn(datapoints, target=datapoints[0][None,:], k=5, distance_metric='l2')
    test_knn(datapoints, target=datapoints[0][None,:], k=5, distance_metric='dot')
    test_knn(datapoints, target=datapoints[0][None,:], k=5, distance_metric='manhattan')

if __name__ == "__main__":
    test()
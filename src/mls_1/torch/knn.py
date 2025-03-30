import torch 

from mls_1.torch.distances import *

def knn(datapoints, target, k=5, distance_metric='l2'):
    distance_function = distance_function_ref[distance_metric]
    distances = distance_function(datapoints, target)
    nearest_neighbors = torch.argsort(distances, axis=0)[:k]
    return nearest_neighbors

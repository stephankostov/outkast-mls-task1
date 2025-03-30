import torch 
import cupy as cp
import numpy as np
import time 
import json 

from mls_1.cupy.distances import *

def knn(datapoints, target, k=5, distance_metric='l2'):
    distance_function = distance_function_ref[distance_metric]
    distances = distance_function(datapoints, target)
    nearest_neighbors = cp.argsort(distances, axis=0)[:k]
    return nearest_neighbors

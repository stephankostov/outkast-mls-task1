
import cupy as cp
import numpy as np

from mls_1.torch.distances import *
from mls_1.utils import *

seed = 1337
np.random.seed(seed)
cp.random.seed(seed)

def distance_cosine_np(X, Y):
    X_normalised = X / np.linalg.norm(X, axis=-1, keepdims=True)
    Y_normalised = Y / np.linalg.norm(Y, axis=-1, keepdims=True)
    cosine_similarity = X_normalised @ np.swapaxes(Y_normalised, -1, -2)
    return 1 - cosine_similarity

def distance_l2_np(X, Y):
    return np.linalg.norm(X - Y, axis=-1)

def distance_dot_np(X, Y):
    return np.dot(X, Y.T)

def distance_manhattan_np(X, Y):
    return np.sum(np.abs(X - Y), axis=-1)

np_distance_function_ref = {
    'cosine': distance_cosine_np,
    'l2': distance_l2_np,
    'dot': distance_dot_np,
    'manhattan': distance_manhattan_np
}

def test_distance_cosine(X, Y): 
    numpy_distance, t = time_function(distance_cosine_np, X, Y)
    X = torch.tensor(X); Y = torch.tensor(Y)
    custom_distance, t = time_function(distance_cosine, X, Y)
    assert np.allclose(numpy_distance, custom_distance), f"Distance mismatch: {numpy_distance} vs {custom_distance}"

def test_distance_l2(X, Y): 
    numpy_distance, t = time_function(distance_l2_np, X, Y)
    X = torch.tensor(X); Y = torch.tensor(Y)
    custom_distance, t = time_function(distance_l2, X, Y)
    assert np.allclose(numpy_distance, custom_distance), f"Distance mismatch: {numpy_distance} vs {custom_distance}"

def test_distance_dot(X, Y):
    numpy_distance, t = time_function(distance_dot_np, X, Y)
    X = torch.tensor(X); Y = torch.tensor(Y)
    custom_distance, t = time_function(distance_dot, X, Y)
    assert np.allclose(numpy_distance, custom_distance), f"Distance mismatch: {numpy_distance} vs {custom_distance}"

def test_distance_manhattan(X, Y):
    numpy_distance, t = time_function(distance_manhattan_np, X, Y)
    X = torch.tensor(X); Y = torch.tensor(Y)
    custom_distance, t = time_function(distance_manhattan, X, Y)
    assert np.allclose(numpy_distance, custom_distance), f"Distance mismatch: {numpy_distance} vs {custom_distance}"

def test():
    
    n_dimensions = int(1e4)
    n_points = 1000
    variance = 10

    X = np.random.rand(1,n_dimensions) * variance
    Y = np.random.rand(1,n_dimensions) * variance
    test_distance_cosine(X, Y)
    test_distance_l2(X, Y)
    test_distance_dot(X, Y)
    test_distance_manhattan(X, Y)
    print("Distance function tests passed for (vec,vec)")
    print()

    X = np.random.rand(1,n_dimensions) * variance
    Y = np.random.rand(n_points,n_dimensions) * variance
    test_distance_cosine(X, Y)
    test_distance_l2(X, Y)
    test_distance_dot(X, Y)
    test_distance_manhattan(X, Y)
    print("Distance function tests passed for (matrix,vec)")
    print()

    X = np.random.rand(n_points,n_dimensions) * variance
    Y = np.random.rand(n_points,n_dimensions) * variance
    test_distance_cosine(X, Y)
    test_distance_l2(X, Y)
    test_distance_dot(X, Y)
    test_distance_manhattan(X, Y)
    print("Distance function tests passed for (matrix,matrix)")
    print()

if __name__ == "__main__":
    test()
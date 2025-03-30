import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))

import numpy as np
import cupy as cp

from distances import test as distances_tests
from knn import test as knn_tests
from kmeans import test as kmeans_tests
from ann import test as ann_tests 

seed = 1337
np.random.seed(seed)
cp.random.seed(seed)


if __name__ == "__main__":
    distances_tests()
    print("Task 1.1 tests passed.")
    knn_tests()
    print("Task 1.2 tests passed.")
    kmeans_tests()
    print("Task 2.1 tests passed.")
    ann_tests()
    print("Task 2.2 tests passed.")
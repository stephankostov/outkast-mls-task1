import cupy as cp

def distance_cosine(X, Y): 
    X_normalised = X / cp.linalg.norm(X, axis=-1, keepdims=True)
    Y_normalised = Y / cp.linalg.norm(Y, axis=-1, keepdims=True)
    cosine_similarity = X_normalised @ cp.swapaxes(Y_normalised, -1, -2)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    return cp.linalg.norm(X - Y, ord=2, axis=-1)

def distance_dot(X, Y):
    return X @ cp.swapaxes(Y, -1, -2)

def distance_manhattan(X, Y):
    return cp.linalg.norm(X - Y, ord=1, axis=-1)

distance_function_ref = {
    'cosine': distance_cosine,
    'l2': distance_l2,
    'dot': distance_dot,
    'manhattan': distance_manhattan
}
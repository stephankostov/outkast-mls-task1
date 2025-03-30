import torch 

def distance_cosine(X, Y):
    X_normalised = X / torch.linalg.norm(X, dim=-1, keepdim=True)
    Y_normalised = Y / torch.linalg.norm(Y, dim=-1, keepdim=True)
    cosine_similarity = X_normalised @ torch.swapaxes(Y_normalised, -1, -2)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    return torch.linalg.norm(X - Y, ord=2, dim=-1)

def distance_dot(X, Y):
    return X @ torch.swapaxes(Y, -1, -2)

def distance_manhattan(X, Y):
    return torch.linalg.norm(X - Y, ord=1, dim=-1)

distance_function_ref = {
    'cosine': distance_cosine,
    'l2': distance_l2,
    'dot': distance_dot,
    'manhattan': distance_manhattan
}
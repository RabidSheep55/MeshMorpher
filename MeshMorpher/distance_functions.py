import numpy as np 

def dist_matrix_euclidian(a, b):
    """
    From two coordinate arrays a, b return matrix
    (len(a), len(b)) of euclidian distances between them
    """
    delta = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(np.power(delta, 2), axis=-1))

def mean_squared_error(a, b):
    """
    Calculate error between two vectors of shape (..., dim)
    mean((ax-bx)^2, (ay-by)^2, ...)
    """
    return np.mean(np.power(a - b, 2), axis=-1)

def error_magnitude(a, b):
    """
    Calculate magnitude of error between two vectors (..., dim) 
    """
    return np.sqrt(np.sum(np.power(a - b, 2), axis=-1))

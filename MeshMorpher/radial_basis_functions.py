import numpy as np

def wendland_c2(dists):
    """
    Apply Wendland's C2 Radial Basis Function to matrix of distances
    """
    phi = np.zeros_like(dists)
    leq1_mask = dists <= 1
    masked_dists = dists[leq1_mask]
    phi[leq1_mask] = (4*masked_dists + 1) * np.power(1 - masked_dists, 4)
    return phi
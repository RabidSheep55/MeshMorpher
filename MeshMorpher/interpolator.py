import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm

from MeshMorpher.config import Config


class RBFInterpolator:
    """ 
    Radial Basis Function Interpolation based on control points
    """

    def __init__(self, cp_coords, cp_disps, config: Config):
        """
        Initialise RBF Morpher from control point coordinates and displacements
        """
        # Store control points coordinates and displacements
        self.cp_coords = cp_coords
        self.cp_disps = cp_disps

        # Store config
        self.config = config

        # Solve weights
        self.solve_weights()

    def solve_weights(self):
        """
        Compute control point weights by solving the linear system of equations
        """
        # Construct matrix of distances between each control point
        distances = self.config.dist_matrix_func(self.cp_coords, self.cp_coords)

        # Wrap distance with radial basis function (scaling with support radius)
        phi = self.config.rb_func(distances / self.config.support_radius)

        # Solve control point weights to match required displacements
        self.weights = linalg.solve(phi, self.cp_disps)

    def batch_interp(self, coords, budget=1e6):
        """
        Perform RBF Interpolation at coords (assumes solved weights)
        Returns displaced coordinates.
        Automatically batches coords if memory budget is exceeded
        """
        weights_size = self.weights.shape[0] * self.weights.shape[1]
        budget = min(budget, weights_size * len(coords))
        batch_size = int(budget // weights_size)

        disps = np.empty(coords.shape, dtype=float)
        for i in tqdm(range(0, len(coords), batch_size), desc='Batch Interp'): 
            disps[i:i+batch_size] = self(coords[i:i+batch_size])
        
        return disps
    
    def __call__(self, coords):
        """
        Perform RBF Interpolation at coords (assumes solved weights)
        Returns displaced coordinates
        """
        # Construct matrix of distances
        distances = self.config.dist_matrix_func(coords, self.cp_coords)

        # Wrap distance with radial basis function (scaling with support radius)
        phi = self.config.rb_func(distances / self.config.support_radius)

        # Scale output of rbf with weights and sum contributions from each cp
        disps = np.dot(phi, self.weights)

        return disps

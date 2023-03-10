import numpy as np
import matplotlib.pyplot as plt 

from ..config import Config
from tqdm import tqdm

class FancyBox():
    """
    **EXPERIMENTAL**
    Attempts to filter out coordinates that are outside a boundary 
    """

    def __init__(self, moving_coords, boundary_coords, config: Config):
        self.config = config

        # Split cp into moving and boundary points 
        self.moving_coords = moving_coords
        self.boundary_coords = boundary_coords

    def filter_points(self, coords, ax=None):
        # Compute minimum distances between coords control points 
        moving_dists = self.config.dist_matrix_func(coords, self.moving_coords)
        boundary_dists = self.config.dist_matrix_func(coords, self.boundary_coords)

        # print(moving_dists.shape, boundary_dists.shape)
        closest_moving_i = np.argmin(moving_dists, axis=1, keepdims=True)
        closest_boundary_i = np.argmin(boundary_dists, axis=1, keepdims=True)

        # Find vectors going from coords to their closest moving or boundary CP 
        vec_moving = self.moving_coords[closest_moving_i[:, 0]] - coords 
        vec_boundary = self.boundary_coords[closest_boundary_i[:, 0]] - coords 

        # if ax: 
        #     selc = np.array([-0.041, 0.889])
        #     isel = np.argmin(np.sum(np.power(coords - selc, 2), axis=-1))

        #     selc = coords[isel]
        #     mov = vec_moving[isel]
        #     stat = vec_boundary[isel]

        #     ax.plot(*np.row_stack([selc, selc+mov]).T)
        #     ax.plot(*np.row_stack([selc, selc+stat]).T)

        # Get vector magnitudes 
        mag_moving = np.take_along_axis(moving_dists, closest_moving_i, axis=1)[:, 0]
        mag_boundary = np.take_along_axis(boundary_dists, closest_boundary_i, axis=1)[:, 0]

        # Compute dot product of vectors 
        dot = np.sum(np.multiply(vec_boundary, vec_moving), axis=1)

        # Coordinate is 'between' closest two points 
        between = dot < 0

        # Coordinate is closer to the moving point 
        closer_to_moving = mag_boundary > mag_moving

        # Within support radius distance from moving point 
        within_moving_range = mag_moving <= self.config.support_radius

        return between | (~between & (closer_to_moving & within_moving_range))
    
    def batch_filter_points(self, coords, budget=1e6): 
        """
        Execute filtering in batches such that len(batch) * len(CP) = budget
        """
        cp_n = int(len(self.boundary_coords) + len(self.moving_coords))
        print(f"Want to filter {len(coords)} coords each considering {cp_n} control points")

        budget = min(budget, cp_n * len(coords))
        batch_size = budget // cp_n

        n_batches = len(coords)/batch_size
        print(f"Chunking input into {n_batches:.2f} batches each with {batch_size} points")

        out = np.empty(len(coords), dtype=bool)
        for i in tqdm(range(0, len(coords), batch_size), desc='Batch Filter'): 
            out[i:i+batch_size] = self.filter_points(coords[i:i+batch_size])

        return out
import numpy as np

from MeshMorpher.config import Config

class FarZeroDisp():
    """
    CP Filter removes zero-displacement points that are 'too' far from moving CP
    to be moved anyways (based on the config support radius)
    """

    def __init__(self, config: Config):
        """
        Initialise with specified configuration
        """
        self.config = config

    def filter_points(self, cp_coords, cp_disps):
        """
        Remove zero displacement control points which are far enough away from 
        moving control points, they won't be moved anyway
        """
        n_cp = len(cp_coords)

        # Extract mask of zerodisp points
        zerodisp_cp_mask = np.all(cp_disps == 0, axis=1)

        # Compute distance of non-moving to moving points
        dists = self.config.dist_matrix_func(cp_coords[zerodisp_cp_mask], cp_coords[~zerodisp_cp_mask])

        # Compute minimum distance for every non-moving point  
        min_dists = np.min(dists, axis=1)

        # Construct index array for every zerodisp point 
        zerodisp_ind = np.arange(n_cp)[zerodisp_cp_mask]

        # Find which zerodisp points are too far 
        far_zerodisp_ind = zerodisp_ind[min_dists > self.config.support_radius]

        # Construct a mask array for filtering these out 
        out_mask = np.ones(n_cp, dtype=bool)
        out_mask[far_zerodisp_ind] = False 

        # print(f"Removing {np.sum(~out_mask)} far away zerodisp points")
        return out_mask
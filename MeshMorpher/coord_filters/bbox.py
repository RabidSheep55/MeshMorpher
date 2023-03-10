import numpy as np

from MeshMorpher.config import Config

class BBox():
    """
    Bounding box filtering for coordinates, initialised from CP coords
    """
    def __init__(self, cp_coords, config: Config):
        self.config = config

        # Calculate bounding box limits in each axis
        self.lims = []
        for ax in range(cp_coords.shape[-1]):
            minimum = np.min(cp_coords[..., ax])
            maximum = np.max(cp_coords[..., ax])

            self.lims += [(minimum - config.support_radius, maximum + config.support_radius)]

    def filter_points(self, coords):
        """
        Return a boolean mask, which filters coords for points inside bbox lims
        """
        mask = np.ones(len(coords), dtype=bool)
        for iax, lim in enumerate(self.lims):
            mask &= np.logical_and(coords[:, iax] >= lim[0], coords[:, iax] <= lim[1])
        return mask
    
    def __call__(self, coords, *arrays):
        """
        Returns coords (and any other arrays supplied) that are in the bbox
        NOTE: if multiple arrays are supplied, their first dimension must be identical 
        """
        mask = self.filter_points(coords)
        
        if len(arrays):
            res = [coords[mask]]
            for a in arrays: 
                res += [a[mask]]

            return res
        else:
            return coords[mask]
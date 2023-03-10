import numpy as np
from tqdm import tqdm

from MeshMorpher.config import Config
from MeshMorpher.interpolator import RBFInterpolator

class GreedyBasic():
    """
    CP Filtering method using a basic, unoptimised Greedy method
    (Inverts the coefficients matrix every iteration)
    """
    def __init__(self, config: Config):
        """
        Initialise class using the Config object 
        """
        # Store inputs
        self.config = config
    
    def filter_points(self, cp_coords, cp_disps):
        """
        Execute the Greedy point selection algorithm, return a mask
        """
        n_cp = len(cp_coords)

        # Initialise mask selecting a subset of control points
        self.selected_cp = np.zeros(n_cp, dtype=bool)
        self.cp_inds = np.arange(n_cp)

        # Start by selecting a control point at random 
        isel = np.random.randint(low=0, high=n_cp-1)
        self.selected_cp[isel] = True 
        
        # Keep adding new points while error is above tolerance
        go = True 
        pbar = tqdm(total=n_cp, desc='Selected Points')
        while go: 
            go, max_error = self.select_next_point(cp_coords, cp_disps) 
            pbar.set_postfix(dict(max_error=max_error), refresh=False)
            pbar.update(int(go))
            

        # We're done, return selection mask 
        return self.selected_cp
        
    
    def select_next_point(self, cp_coords, cp_disps):
        """
        Find the next best control point to add, 
        return True if point was added
        """
        # If we've selected all points except one, return full mask 
        if np.sum(self.selected_cp) == len(cp_coords) - 1:
            self.selected_cp[:] = True
            return False, 0

        # Extract selected points 
        cur_cp_coords = cp_coords[self.selected_cp]
        cur_cp_disps = cp_disps[self.selected_cp]

        # Solve the RBF weights using the selected control points 
        interp = RBFInterpolator(cur_cp_coords, cur_cp_disps, self.config)

        # Calculate displacement at unselected control points 
        unsel_cp_coords = cp_coords[~self.selected_cp]
        unsel_interp_disps = interp(unsel_cp_coords)
        
        # Compute error with known desired displacement at those points
        unsel_cp_disps = cp_disps[~self.selected_cp]
        error = self.config.err_func(unsel_cp_disps, unsel_interp_disps)

        # If maximum error is below requested tolerance, stop
        max_error = np.max(error)
        if max_error < self.config.tol: 
            return False, max_error
        
        # Determine control point with max error 
        imax = np.argmax(error)

        # Relate index to all control point 
        unsel_cp_inds = self.cp_inds[~self.selected_cp]
        icp = unsel_cp_inds[imax]

        # Add the point to the selection 
        self.selected_cp[icp] = True 
        return True, max_error
import numpy as np
from tqdm import tqdm
import scipy.linalg as linalg

from ..config import Config


class GreedyOpt():
    """
    This Greedy filtering method optimises the matrix inversion step as described in
    https://www.mdpi.com/2076-3417/9/6/1141
    """

    def __init__(self, config: Config):
        """
        Initialise class using the Config object 
        """
        self.config = config

    def filter_points(self, cp_coords, cp_disps):
        """
        Filter points using an optimised greedy method
        """
        # Useful vars
        n_cp = len(cp_coords)
        cp_inds = np.arange(n_cp, dtype=int)

        # Initialise solution
        selected_mask = np.zeros(n_cp, dtype=bool)
        selected_inds = np.array([], dtype=int)

        # Select a point at random for the first iteration
        isel = np.random.randint(low=0, high=n_cp - 1)
        max_error = np.inf
        first_iter = True

        # Tracking progress
        pbar = tqdm(desc='Selected CP', total=n_cp)

        # Keep adding points while the max_error is above the tolerance
        while (max_error >= self.config.tol) and len(
                selected_inds) < n_cp:
            # Update structures tracking the selected cp
            selected_mask[isel] = True
            selected_inds = np.append(selected_inds, [isel])
            pbar.update(1)

            # Extract selected and unselected coords
            cur_cp_coords = cp_coords[selected_inds]
            unsel_cp_coords = cp_coords[~selected_mask]

            if first_iter:
                # For the first iter, we don't have any old_phi_inv, calculate it
                distances = self.config.dist_matrix_func(cur_cp_coords, cur_cp_coords)
                phi = self.config.rb_func(distances / self.config.support_radius)
                phi_inv = linalg.inv(phi)

                first_iter = False

            else:
                # Yield the new inverted phi matrix from the previous one
                phi_inv = self.get_phi_inv(old_phi_inv, cur_cp_coords)

            # Calculate new control point weights
            cur_cp_disps = cp_disps[selected_inds]
            cur_cp_weights = np.dot(phi_inv, cur_cp_disps)

            # Interpolate rbf at unselected control points
            unsel_cp_interp_disps = self.interp_RBF(unsel_cp_coords,
                                                    cur_cp_coords,
                                                    cur_cp_weights)

            # Compute error with actual desired displacement at unselected points
            unsel_cp_disps = cp_disps[~selected_mask]
            error = self.config.err_func(unsel_cp_disps, unsel_cp_interp_disps)

            # Recalculate maximum error
            max_error = np.max(error)
            pbar.set_postfix(dict(max_error=max_error))
            imax = np.argmax(error)  # This indexes unsel_cp_disps

            # imax indexes unsel_cp_disps, relate back to index in all cp
            unsel_cp_inds = cp_inds[~selected_mask]
            isel = unsel_cp_inds[imax]

            old_phi_inv = phi_inv

        # We're done
        return selected_mask

    def interp_RBF(self, coords, cp_coords, cp_weights):
        """
        Perform RBF interpolation at coords using pre-calculated weights
        """
        # Construct matrix of distances
        distances = self.config.dist_matrix_func(coords, cp_coords)

        # Wrap distance with radial basis function (scaling with support radius)
        phi = self.config.rb_func(distances / self.config.support_radius)

        # Scale output of rbf with weights and sum contributions from each cp
        disps = np.dot(phi, cp_weights)

        return disps

    def get_phi_inv(self, old_phi_inv, cp_coords):
        """
        Calculate new inverted matrix from previous value, and currently selected cp
        NOTE: cp_coords needs to be updated by appending only
        """
        # print(f"[PHI] Old phi_inv shape {old_phi_inv.shape}, current cp_coords {cp_coords.shape}")

        # Get coordinate of last added point
        new_coord = cp_coords[-1]

        # Get distances between it and previously selected coords
        new_dists = self.config.dist_matrix_func(cp_coords[:-1],
                                          new_coord[np.newaxis, :])
        r = self.config.rb_func(new_dists / self.config.support_radius)
        # print(f"[PHI] r.shape : {r.shape}")

        # Construct beta and b matrices from previously inverted phi
        beta = -np.dot(old_phi_inv, r)
        b = 1 / (self.config.rb_zeroval -
                 np.linalg.multi_dot([r.T, old_phi_inv, r]))
        # print(f"[PHI] beta.shape : {beta.shape}")

        # Create a new inverted phi from the old
        new_phi_inv = np.block([
            [old_phi_inv + b * beta * beta.T, b * beta],
            [b * beta.T, b],
        ])

        # print(f"[PHI] New phi shape {new_phi_inv.shape}")

        return new_phi_inv

from .radial_basis_functions import wendland_c2
from .distance_functions import error_magnitude, dist_matrix_euclidian
from typing import NamedTuple, Callable


class Config(NamedTuple):
    """
    Default configuration class used by all classes in this package
    Contains specified thresholds and default functions
    """
    # Store thresholds and parameters
    tol: float
    support_radius: float 

    # Distance func used to compute distance between coords
    dist_matrix_func: Callable = dist_matrix_euclidian

    # Radial basis function
    rb_func: Callable = wendland_c2
    rb_zeroval: float = 1.

    # Error function used to compare displacements
    err_func: Callable = error_magnitude


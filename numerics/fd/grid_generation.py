import math
import numpy as np


def equidistant(x_min: float,
                x_max: float,
                n_states: int) -> np.ndarray:
    """Equidistant grid in one dimension.

    Args:
        x_min: "Left" boundary of domain.
        x_max: "Right" boundary of domain.
        n_states: Number of grid points.

    Returns:
        Equidistant grid.
    """
    dx = (x_max - x_min) / (n_states - 1)
    return dx * np.arange(n_states) + x_min


def hyperbolic(x_min: float,
               x_max: float,
               n_states: int,
               x_center: float,
               shape: float = None) -> (np.ndarray, np.ndarray):
    """Non-equidistant grid in one dimension.

    Hyperbolic sine transformation. See In 'T Hout and Foulon (2010).

    Args:
        x_min: "Left" boundary of domain.
        x_max: "Right" boundary of domain.
        n_states: Number of grid points.
        x_center: "Center" of non-equidistant grid.
        shape: Controls the fraction of grid points in the neighbourhood
            of x_center. Default is x_center / 5.

    Returns:
        Non-equidistant grid.
    """
    if x_center < x_min or x_center > x_max:
        raise ValueError("Grid 'center' is not contained in domain.")
    if not shape:
        shape = x_center / 5
    lower = math.asinh(-(x_center - x_min) / shape)
    delta = (math.asinh((x_max - x_center) / shape) - lower) / (n_states - 1)
    grid = delta * np.arange(n_states) + lower
    return grid, x_center + shape * np.sinh(grid)

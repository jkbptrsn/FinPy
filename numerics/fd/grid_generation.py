import numpy as np


def equidistant(x_min: float,
                x_max: float,
                x_states: int):
    """Equidistant grid in 1-dimension.

    Args:
        x_min: Left boundary of domain.
        x_max: Right boundary of domain.
        x_states: Number of grid points.

    Returns:
        Equidistant grid.
    """
    dx = (x_max - x_min) / (x_states - 1)
    return dx * np.arange(x_states) + x_min


def non_equidistant(x_min: float,
                    x_max: float,
                    x_states: int,
                    type_: str = "hyperbolic"):
    """Equidistant grid in 1-dimension.

    Args:
        x_min: Left boundary of domain.
        x_max: Right boundary of domain.
        x_states: Number of grid points.
        type_: Type of grid. TODO: Default is hyperbolic.

    Returns:
        Non-equidistant grid.
    """
    pass

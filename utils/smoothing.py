import numpy as np


def smoothing_1d(grid: np.ndarray,
                 function: np.ndarray) -> np.ndarray:
    """'Box smoothing' of discrete function on non-equidistant grid.

    Smoothing of discrete function. The method is similar to box
    smoothing. See L.B.G. Andersen & V.V. Piterbarg 2010, section 23.3.

    Args:
        grid: Discrete grid.
        function: Discrete function.

    Returns:
        Smooth function.
    """
    smooth = np.zeros(function.size)
    smooth[0] = function[0]
    smooth[-1] = function[-1]
    # Linear interpolation of function at midpoints.
    f_midpoint = (function[1:] + function[:-1]) / 2
    step_size = np.diff(grid)
    # Loop over interior points.
    for idx in range(grid.size - 2):
        average_step_size = (step_size[idx + 1] + step_size[idx]) / 2
        weight_left = step_size[idx + 1] / (2 * average_step_size)
        weight_right = step_size[idx] / (2 * average_step_size)
        f_left = weight_left * f_midpoint[idx]
        f_right = weight_right * f_midpoint[idx + 1]
        smooth[idx + 1] = f_left + f_right
    return smooth


def smoothing_payoff_1d(grid: np.ndarray,
                        instrument) -> np.ndarray:
    """Box smoothing of payoff function on non-equidistant grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, section 23.3.

    Args:
        grid: Discrete grid.
        instrument: Instrument object with 'payoff' method.

    Returns:
        Smooth payoff.
    """
    # Extended grid with midpoints.
    extended_grid = np.zeros(2 * grid.size - 1)
    extended_grid[::2] = grid
    for idx in range(grid.size - 1):
        extended_grid[2 * idx + 1] = (grid[idx] + grid[idx + 1]) / 2
    payoff = instrument.payoff(extended_grid)
    smooth = np.zeros(grid.size)
    smooth[0] = payoff[0]
    smooth[-1] = payoff[-1]
    step_size = np.diff(grid)
    # Loop over interior points.
    for idx in range(grid.size - 2):
        average_step_size = (step_size[idx + 1] + step_size[idx]) / 2
        weight_left = step_size[idx + 1] / (2 * average_step_size)
        weight_right = step_size[idx] / (2 * average_step_size)
        f_left = weight_left * payoff[2 * idx + 1]
        f_right = weight_right * payoff[2 * idx + 3]
        smooth[idx + 1] = f_left + f_right
    return smooth

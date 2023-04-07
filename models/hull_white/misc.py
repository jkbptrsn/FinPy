import math
import numpy as np

from utils import misc


def setup_int_grid(event_grid: np.ndarray,
                   int_step_size: float) -> (np.ndarray, np.ndarray):
    """Set up time grid for numerical integration."""
    # Assume that the first event is the initial time point on the
    # integration grid.
    int_grid = np.array(event_grid[0])
    # The first event has index zero on the integration grid.
    int_event_idx = np.array(0)
    # Step size between two adjacent events.
    step_size_grid = np.diff(event_grid)
    for idx, step_size in enumerate(step_size_grid):
        # Number of integration steps.
        steps = math.floor(step_size / int_step_size)
        initial_date = event_grid[idx]
        if steps == 0:
            grid = np.array(initial_date + step_size)
        else:
            grid = int_step_size * np.arange(1, steps + 1) + initial_date
            diff_step = step_size - steps * int_step_size
            if diff_step > 1.0e-12:
                grid = np.append(grid, grid[-1] + diff_step)
        int_grid = np.append(int_grid, grid)
        int_event_idx = np.append(int_event_idx, grid.size)
    int_event_idx = np.cumsum(int_event_idx)
    return int_grid, int_event_idx


def y_constant(kappa: float,
               vol: float,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    Constant kappa and vol. See L.B.G. Andersen & V.V. Piterbarg 2010,
    proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        y-function.
    """
    two_kappa = 2 * kappa
    return vol ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa


def y_piecewise(kappa: float,
                vol: np.ndarray,
                event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    Constant kappa and piecewise constant vol. See
    L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        y-function.
    """
    y_return = np.zeros(event_grid.size)
    two_kappa = 2 * kappa
    for idx in range(1, event_grid.size):
        event_filter = event_grid <= event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        delta_t = event_grid[idx] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa
        y_return[idx] = y.sum()
    return y_return


def y_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step: np.ndarray,
              vol_int_grid: np.ndarray,
              event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate y-function on event grid and integration grid.

    General time-dependence of kappa and vol. See
    L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_int_grid: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        y-function.
    """
    # Calculation of y-function on integration grid.
    y_ig = np.zeros(int_grid.size)
    for idx in range(1, int_grid.size):
        # int_u^t_{idx} kappa_s ds.
        int_kappa = int_kappa_step[:idx + 1]
        # Cumulative sum from "right to left"
        int_kappa = np.cumsum(int_kappa[::-1])[::-1]
        int_kappa[:-1] = int_kappa[1:]
        int_kappa[-1] = 0
        # Integrand in expression for y.
        integrand = np.exp(-2 * int_kappa) * vol_int_grid[:idx + 1] ** 2
        y_ig[idx] = np.sum(misc.trapz(int_grid[:idx + 1], integrand))
    # Save y-function on event grid.
    y_eg = np.zeros(event_grid.size)
    for event_idx, int_idx in enumerate(int_event_idx):
        y_eg[event_idx] = y_ig[int_idx]
    return y_eg, y_ig

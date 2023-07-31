import math
import typing

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

from utils.global_types import Instrument
from utils import misc


def setup_model_parameters(inst):
    """Set up model parameters on event and integration grids."""
    # Kappa interpolated on event grid.
    inst.kappa_eg = inst.kappa.interpolation(inst.event_grid)
    # Vol interpolated on event grid.
    inst.vol_eg = inst.vol.interpolation(inst.event_grid)
    # Discount curve interpolated on event grid.
    inst.discount_curve_eg = inst.discount_curve.interpolation(inst.event_grid)

    # Instantaneous forward rate on event grid.
    # TODO: Test accuracy
    log_discount = np.log(inst.discount_curve_eg)
    smoothing = 0
    log_discount_spline = \
        UnivariateSpline(inst.event_grid, log_discount, s=smoothing)
    forward_rate = log_discount_spline.derivative()
    inst.forward_rate_eg = -forward_rate(inst.event_grid)

    # Kappa and vol are constant.
    if inst.time_dependence == "constant":
        # Integration of kappa on event grid.
        inst.int_kappa_eg = inst.kappa_eg[0] * inst.event_grid
        # G-function, G(0,t), on event grid.
        inst.g_eg = g_constant(inst.kappa_eg[0], inst.event_grid)
        # y-function on event grid.
        inst.y_eg = y_constant(inst.kappa_eg[0],
                               inst.vol_eg[0],
                               inst.event_grid)
    # Kappa is constant and vol is piecewise constant.
    elif inst.time_dependence == "piecewise":
        # Integration of kappa on event grid.
        inst.int_kappa_eg = inst.kappa_eg[0] * inst.event_grid
        # G-function, G(0,t), on event grid.
        inst.g_eg = g_constant(inst.kappa_eg[0], inst.event_grid)
        # y-function on event grid.
        inst.y_eg = y_piecewise(inst.kappa_eg[0], inst.vol_eg, inst.event_grid)
    # Kappa and vol have general time dependence.
    elif inst.time_dependence == "general":
        # Kappa interpolated on integration grid.
        inst.kappa_ig = inst.kappa.interpolation(inst.int_grid)
        # Vol interpolated on integration grid.
        inst.vol_ig = inst.vol.interpolation(inst.int_grid)
        # Step-wise integration of kappa on integration grid.
        inst.int_kappa_step_ig = \
            np.append(0, misc.trapz(inst.int_grid, inst.kappa_ig))
        # Integration of kappa on event grid.
        inst.int_kappa_eg = np.zeros(inst.event_grid.size)
        for event_idx, int_idx in enumerate(inst.int_event_idx):
            inst.int_kappa_eg[event_idx] = \
                np.sum(inst.int_kappa_step_ig[:int_idx + 1])
        # G-function, G(0,t), on event grid.
        inst.g_eg = g_general(inst.int_grid,
                              inst.int_event_idx,
                              inst.int_kappa_step_ig,
                              inst.event_grid)
        # y-function on event grid.
        inst.y_eg, _ = y_general(inst.int_grid,
                                 inst.int_event_idx,
                                 inst.int_kappa_step_ig,
                                 inst.vol_ig,
                                 inst.event_grid)
    else:
        raise ValueError(f"Time dependence is unknown: {inst.time_dependence}")


def integration_grid(event_grid: np.ndarray,
                     int_dt: float) -> (np.ndarray, np.ndarray):
    """Set up time grid for numerical integration.

    Args:
        event_grid: Event dates as year fractions from as-of date.
        int_dt: Integration step size.

    Returns:
        Integration time grid.
    """
    # Assume that the first event is the initial time point on the
    # integration grid.
    int_grid = np.array(event_grid[0])
    # The first event has index zero on the integration grid.
    int_event_idx = np.array(0)
    # Step size between two adjacent events.
    step_size_grid = np.diff(event_grid)
    for idx, step_size in enumerate(step_size_grid):
        # Number of integration steps.
        steps = math.floor(step_size / int_dt)
        initial_date = event_grid[idx]
        if steps == 0:
            grid = np.array(initial_date + step_size)
        else:
            grid = int_dt * np.arange(1, steps + 1) + initial_date
            diff_step = step_size - steps * int_dt
            if diff_step > 1.0e-12:
                grid = np.append(grid, grid[-1] + diff_step)
        int_grid = np.append(int_grid, grid)
        int_event_idx = np.append(int_event_idx, grid.size)
    int_event_idx = np.cumsum(int_event_idx)
    return int_grid, int_event_idx


def g_function(maturity_idx: int,
               g_eg: np.ndarray,
               int_kappa_eg: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(t,t_maturity), on event grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, remark 10.1.9.

    Args:
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.
        int_kappa_eg: Integral of kappa on event grid.

    Returns:
        G-function.
    """
    return (g_eg[maturity_idx] - g_eg) * np.exp(int_kappa_eg)


def g_constant(kappa: float,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(0,t), on event grid.

    The speed of mean reversion is constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        G-function.
    """
    return (1 - np.exp(-kappa * event_grid)) / kappa


def g_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step_ig: np.ndarray,
              event_grid: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(0,t), on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        G-function.
    """
    # Calculation of G-function, G(0,t), on integration grid.
    g_ig = np.zeros(int_grid.size)
    for idx in range(1, int_grid.size):
        # Slice of integration grid.
        int_grid_slice = int_grid[:idx + 1]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = int_kappa_step_ig[:idx + 1]
        # Integrand in expression for G-function, G(0,t).
        integrand = np.exp(-np.cumsum(int_kappa))
        g_ig[idx] = np.sum(misc.trapz(int_grid_slice, integrand))
    # Save G-function, G(0,t), on event grid.
    g_eg = np.zeros(event_grid.size)
    for event_idx, int_idx in enumerate(int_event_idx):
        g_eg[event_idx] = g_ig[int_idx]
    return g_eg


def y_constant(kappa: float,
               vol: float,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    The speed of mean reversion and volatility are constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        y-function.
    """
    two_kappa = 2 * kappa
    return vol ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa


def int_y_constant(kappa: float,
                   vol: float,
                   event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    The speed of mean reversion and volatility are constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010c

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Integral" of y-function.
    """
    exp_kappa_1 = np.exp(-2 * kappa * event_grid[1:])
    exp_kappa_2 = np.exp(-kappa * np.diff(event_grid))
    event_grid_sum = event_grid[1:] + event_grid[:-1]
    exp_kappa_3 = np.exp(-kappa * event_grid_sum)
    integral = np.zeros(event_grid.size)
    integral[1:] = vol ** 2 * (1 + exp_kappa_1
                               - exp_kappa_2 - exp_kappa_3) / (2 * kappa ** 2)
    return integral


def int_int_y_constant(kappa: float,
                       vol: float,
                       event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    The speed of mean reversion and volatility are constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Double integral" of y-function.
    """
    exp_kappa_1 = (np.exp(-2 * kappa * event_grid[:-1])
                   - np.exp(-2 * kappa * event_grid[1:])) / 2
    exp_kappa_2 = np.exp(-kappa * np.diff(event_grid)) - 1
    event_grid_sum = event_grid[1:] + event_grid[:-1]
    exp_kappa_3 = \
        np.exp(-kappa * event_grid_sum) - np.exp(-2 * kappa * event_grid[:-1])
    integral = np.zeros(event_grid.size)
    integral[1:] = \
        vol ** 2 * (kappa * np.diff(event_grid) + exp_kappa_1
                    + exp_kappa_2 + exp_kappa_3) / (2 * kappa ** 3)
    return integral


def y_piecewise(kappa: float,
                vol: np.ndarray,
                event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        y-function.
    """
    y_function = np.zeros(event_grid.size)
    two_kappa = 2 * kappa
    for event_idx in range(1, event_grid.size):
        # See notes for "less than or equal to".
        event_filter = event_grid <= event_grid[event_idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        delta_t = event_grid[event_idx] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa
        y_function[event_idx] = y.sum()
    return y_function


def int_y_piecewise(kappa: float,
                    vol: np.ndarray,
                    event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Integral" of y-function.
    """
    integral = np.zeros(event_grid.size)
    two_kappa = 2 * kappa
    two_kappa_sq = 2 * kappa ** 2
    for idx in range(1, event_grid.size):
        # See notes for "less than".
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        # First term.
        delta_t = event_grid[idx] - vol_times
        y = np.exp(-two_kappa * delta_t[:-1]) \
            - np.exp(-two_kappa * delta_t[1:])
        y *= vol_values[:-1] ** 2 / two_kappa_sq
        integral[idx] += y.sum()

        # Second term.
        # TODO: Make unit tests.
#        delta_t = event_grid[idx] - event_grid[idx - 1]
        delta_t = event_grid[idx] - vol_times[-1]
        y = 1 + math.exp(-two_kappa * delta_t)
        y *= vol_values[-1] ** 2 / two_kappa_sq
        integral[idx] += y

        # Third term.
        delta_t = event_grid[idx] + event_grid[idx - 1] - 2 * vol_times
        y = np.exp(-kappa * delta_t[:-1]) - np.exp(-kappa * delta_t[1:])
        y *= vol_values[:-1] ** 2 / two_kappa_sq
        integral[idx] -= y.sum()
        # Fourth term.
        delta_t = event_grid[idx] - event_grid[idx - 1]
        y = 2 * math.exp(-kappa * delta_t)
        y *= vol_values[-1] ** 2 / two_kappa_sq
        integral[idx] -= y
    return integral


def int_int_y_piecewise(kappa: float,
                        vol: np.ndarray,
                        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Double integral" of y-function.
    """
    integral = np.zeros(event_grid.size)
    two_kappa = 2 * kappa
    two_kappa_cubed = 2 * kappa ** 3
    for idx in range(1, event_grid.size):
        # See notes for "less than".
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        # First term.
        delta_t = event_grid[idx] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] += y.sum()
        delta_t = event_grid[idx] - vol_times[-1]
        y = two_kappa * event_grid[idx] - math.exp(-two_kappa * delta_t)
        y *= vol_values[-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] += y
        # Second term.
        delta_t = event_grid[idx] + event_grid[idx - 1] - 2 * vol_times
        y = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[idx] -= y.sum()

        # TODO: Make unit tests.
#        delta_t = event_grid[idx] - event_grid[idx - 1]
#        y = math.exp(-kappa * delta_t)
#        delta_t = event_grid[idx] + event_grid[idx - 1] - 2 * vol_times[-1]
#        y += math.exp(-kappa * delta_t)
        delta_t = event_grid[idx] - event_grid[idx - 1]
        y = 2 * math.exp(-kappa * delta_t)

        y *= vol_values[-1] ** 2 / two_kappa_cubed
        integral[idx] += y
        # Third term.
        delta_t = event_grid[idx - 1] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] -= y.sum()
        y = two_kappa * event_grid[idx - 1] - 1
        y *= vol_values[-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] -= y
        # Fourth term.
        delta_t = event_grid[idx - 1] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[idx] += y.sum()
        y = 2 * vol_values[-1] ** 2 / two_kappa_cubed
        integral[idx] -= y
    return integral


def y_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step_ig: np.ndarray,
              vol_ig: np.ndarray,
              event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate y-function on event and integration grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        y-function.
    """
    # Calculation of y-function on integration grid.
    y_ig = np.zeros(int_grid.size)
    for idx in range(1, int_grid.size):
        # int_u^t_{idx} kappa_s ds.
        int_kappa = int_kappa_step_ig[:idx + 1]
        # Cumulative sum from "right to left".
        int_kappa = np.flip(np.cumsum(np.flip(int_kappa)))
        # Shift to the left.
        int_kappa[:-1] = int_kappa[1:]
        int_kappa[-1] = 0
        # Integrand in expression for y.
        integrand = np.exp(-2 * int_kappa) * vol_ig[:idx + 1] ** 2
        y_ig[idx] = np.sum(misc.trapz(int_grid[:idx + 1], integrand))
    # Save y-function on event grid.
    y_eg = np.zeros(event_grid.size)
    for event_idx, int_idx in enumerate(int_event_idx):
        y_eg[event_idx] = y_ig[int_idx]
    return y_eg, y_ig


def int_y_general(int_grid: np.ndarray,
                  int_event_idx: np.ndarray,
                  int_kappa_step_ig: np.ndarray,
                  vol_ig: np.ndarray,
                  event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Integral" of y-function.
    """
    _, y_ig = y_general(int_grid, int_event_idx, int_kappa_step_ig,
                        vol_ig, event_grid)
    # Calculation of "integral" of y-function on event grid.
    integral = np.zeros(event_grid.size)
    for event_idx in range(1, event_grid.size):
        # Integration indices of two adjacent events.
        idx1 = int_event_idx[event_idx - 1]
        idx2 = int_event_idx[event_idx] + 1
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx1:idx2]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = np.append(int_kappa_step_ig[idx1 + 1:idx2], 0)
        int_kappa = np.flip(np.cumsum(np.flip(int_kappa)))
        integrand = np.exp(-int_kappa) * y_ig[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return integral


def int_int_y_general(int_grid: np.ndarray,
                      int_event_idx: np.ndarray,
                      int_kappa_step_ig: np.ndarray,
                      vol_ig: np.ndarray,
                      event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Double integral" of y-function.
    """
    _, y_ig = y_general(int_grid, int_event_idx, int_kappa_step_ig,
                        vol_ig, event_grid)
    # Calculation of "double integral" of y-function on event grid.
    integral = np.zeros(event_grid.size)
    for event_idx in range(1, event_grid.size):
        # Integration indices of two adjacent events.
        idx1 = int_event_idx[event_idx - 1]
        idx2 = int_event_idx[event_idx] + 1
        # Double time integral in Eq. (10.42).
        inner_integral = np.array(0)
        for idx in range(idx1 + 1, idx2):
            int_grid_tmp = int_grid[idx1:idx + 1]
            int_kappa_tmp = np.append(int_kappa_step_ig[idx1 + 1:idx + 1], 0)
            int_kappa_tmp = np.flip(np.cumsum(np.flip(int_kappa_tmp)))
            integrand = np.exp(-int_kappa_tmp) * y_ig[idx1:idx + 1]
            inner_integral = \
                np.append(inner_integral,
                          np.sum(misc.trapz(int_grid_tmp, integrand)))
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, inner_integral))
    return integral


def alpha_constant(kappa: float,
                   vol: float,
                   event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    The speed of mean reversion and volatility are constant.

    The function doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        alpha-function.
    """
    return vol ** 2 * (1 - np.exp(-kappa * event_grid)) ** 2 / (2 * kappa ** 2)


def int_alpha_constant(kappa: float,
                       vol: float,
                       event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    The speed of mean reversion and volatility are constant.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        Integral of alpha-function.
    """
    exp_kappa_1 = np.exp(-2 * kappa * event_grid[:-1]) \
        - np.exp(-2 * kappa * event_grid[1:])
    exp_kappa_2 = np.exp(-kappa * event_grid[1:]) \
        - np.exp(-kappa * event_grid[:-1])
    integral = np.zeros(event_grid.size)
    integral[1:] = \
        vol ** 2 * (2 * kappa * np.diff(event_grid)
                    + exp_kappa_1 + 4 * exp_kappa_2) / (4 * kappa ** 3)
    return integral


def alpha_general(int_grid: np.ndarray,
                  int_event_idx: np.ndarray,
                  int_kappa_step_ig: np.ndarray,
                  vol_ig: np.ndarray,
                  event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    The function doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        alpha-function.
    """
    _, y_ig = y_general(int_grid, int_event_idx, int_kappa_step_ig,
                        vol_ig, event_grid)
    integral = np.zeros(event_grid.size)
    for event_idx in range(1, event_grid.size):
        # Integration index of event.
        idx = int_event_idx[event_idx] + 1
        # Slice of integration grid.
        int_grid_tmp = int_grid[:idx]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = np.append(int_kappa_step_ig[1:idx], 0)
        int_kappa = np.flip(np.cumsum(np.flip(int_kappa)))
        integrand = np.exp(-int_kappa) * y_ig[:idx]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return integral


def int_alpha_general(int_grid: np.ndarray,
                      int_event_idx: np.ndarray,
                      int_kappa_step_ig: np.ndarray,
                      vol_ig: np.ndarray,
                      event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        Integral of alpha-function.
    """
    _, y_ig = y_general(int_grid, int_event_idx, int_kappa_step_ig,
                        vol_ig, event_grid)
    integral = np.zeros(event_grid.size)
    for event_idx in range(1, event_grid.size):
        # Integration indices of two adjacent events.
        idx1 = int_event_idx[event_idx - 1]
        idx2 = int_event_idx[event_idx] + 1
        inner_integral = np.array(0)
        for idx in range(idx1 + 1, idx2):
            int_grid_tmp = int_grid[:idx + 1]
            int_kappa_tmp = np.append(int_kappa_step_ig[1:idx + 1], 0)
            int_kappa_tmp = np.flip(np.cumsum(np.flip(int_kappa_tmp)))
            integrand = np.exp(-int_kappa_tmp) * y_ig[:idx + 1]
            inner_integral = \
                np.append(inner_integral,
                          np.sum(misc.trapz(int_grid_tmp, integrand)))
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, inner_integral))
    return integral


def alpha_piecewise(kappa: float,
                    vol: np.ndarray,
                    event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    The function doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        alpha-function.
    """
    two_kappa_sq = 2 * kappa ** 2
    sum_array = np.zeros(event_grid.size)
    for idx in range(1, event_grid.size):
        # See notes for "less than".
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        # Upper limit, see notes.
        delta_t = event_grid[idx] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[:-1]) - np.exp(-kappa * delta_t[1:])
        tmp *= vol_values[:-1] ** 2 / two_kappa_sq
        sum_array[idx] += tmp.sum()
        delta_t = event_grid[idx] - 2 * vol_times[-1]
        tmp = math.exp(kappa * event_grid[idx]) + math.exp(-kappa * delta_t)
        sum_array[idx] += vol_values[-1] ** 2 * tmp / two_kappa_sq
        # Lower limit, see notes.
        delta_t = event_grid[idx - 1] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[:-1]) - np.exp(-kappa * delta_t[1:])
        tmp *= vol_values[:-1] ** 2 / two_kappa_sq
        sum_array[idx] -= tmp.sum()
        delta_t = event_grid[idx - 1] - 2 * vol_times[-1]
        tmp = \
            math.exp(kappa * event_grid[idx - 1]) + math.exp(-kappa * delta_t)
        sum_array[idx] -= vol_values[-1] ** 2 * tmp / two_kappa_sq
    return np.exp(-kappa * event_grid) * np.cumsum(sum_array)


def int_alpha_piecewise(kappa: float,
                        vol: np.ndarray,
                        event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        Integral of alpha-function.
    """
    two_kappa = 2 * kappa
    two_kappa_cubed = 2 * kappa ** 3
    sum_array = np.zeros(event_grid.size)
    for idx in range(1, event_grid.size):
        # See notes for "less than".
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        # Upper limit, see notes.
        delta_t = event_grid[idx] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        tmp *= vol_values[:-1] ** 2 / two_kappa_cubed
        sum_array[idx] += tmp.sum()
        delta_t = event_grid[idx] - 2 * vol_times[-1]
        tmp = math.exp(kappa * event_grid[idx]) + math.exp(-kappa * delta_t)
        sum_array[idx] -= vol_values[-1] ** 2 * tmp / two_kappa_cubed
        # Lower limit, see notes.
        delta_t = event_grid[idx - 1] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        tmp *= vol_values[:-1] ** 2 / two_kappa_cubed
        sum_array[idx] -= tmp.sum()
        delta_t = event_grid[idx - 1] - 2 * vol_times[-1]
        tmp = \
            math.exp(kappa * event_grid[idx - 1]) + math.exp(-kappa * delta_t)
        sum_array[idx] += vol_values[-1] ** 2 * tmp / two_kappa_cubed
    integral = np.zeros(event_grid.size)
    for idx in range(1, event_grid.size):
        # See notes for "less than".
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]

        # First kind of integral, see notes.
        factor = math.exp(-kappa * event_grid[idx]) \
            - math.exp(-kappa * event_grid[idx - 1])
        integral[idx] = np.sum(factor * sum_array[:idx])

        # Second kind of integral, see notes.
        # Upper-upper limit, see notes.
        delta_t = event_grid[idx] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] += y.sum()
        delta_t = event_grid[idx] - event_grid[idx - 1]
        y = math.exp(-two_kappa * delta_t) - two_kappa * event_grid[idx]
        y *= vol_values[-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] -= y

        # Upper-lower limit, see notes.
        delta_t = event_grid[idx - 1] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] -= y.sum()
        y = 1 - two_kappa * event_grid[idx - 1]
        y *= vol_values[-1] ** 2 / (2 * two_kappa_cubed)
        integral[idx] += y

        # Lower-upper limit, see notes.
        delta_t = event_grid[idx] + event_grid[idx - 1] - 2 * vol_times
        y = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[idx] -= y.sum()
        delta_t = event_grid[idx] - event_grid[idx - 1]
        y = math.exp(-kappa * delta_t)
        y *= 2 * vol_values[-1] ** 2 / two_kappa_cubed
        integral[idx] += y

        # Lower-lower limit, see notes.
        delta_t = event_grid[idx - 1] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[idx] += y.sum()
        integral[idx] -= 2 * vol_values[-1] ** 2 / two_kappa_cubed
    return integral


########################################################################


def v_constant(kappa: float,
               vol: float,
               expiry_idx: int,
               maturity_idx: int,
               g_eg: np.ndarray,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate v-function on event grid until expiry.

    The speed of mean reversion is constant and volatility is constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        v-function.
    """
    two_kappa = 2 * kappa
    exp_kappa1 = math.exp(two_kappa * event_grid[expiry_idx])
    exp_kappa2 = np.exp(two_kappa * event_grid[:expiry_idx + 1])
    return vol ** 2 * (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2 \
        * (exp_kappa1 - exp_kappa2) / two_kappa


def dv_dt_constant(kappa: float,
                   vol: float,
                   expiry_idx: int,
                   maturity_idx: int,
                   g_eg: np.ndarray,
                   event_grid: np.ndarray) -> np.ndarray:
    """Calculate 1st order time derivative of v-function.

    The speed of mean reversion is constant and volatility is constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        1st order time derivative of v-function.
    """
    return -vol ** 2 * (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2 \
        * np.exp(2 * kappa * event_grid[:expiry_idx + 1])


def v_piecewise(kappa: float,
                vol: np.ndarray,
                expiry_idx: int,
                maturity_idx: int,
                g_eg: np.ndarray,
                event_grid: np.ndarray) -> np.ndarray:
    """Calculate v-function on event grid until expiry.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        v-function.
    """
    two_kappa = 2 * kappa
    factor = (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2
    v_return = np.zeros(expiry_idx + 1)
    for idx in range(expiry_idx + 1):
        vol_times = event_grid[idx:expiry_idx + 1]
        vol_values = vol[idx:expiry_idx + 1]
        v = np.exp(two_kappa * vol_times[1:]) \
            - np.exp(two_kappa * vol_times[:-1])
        v *= vol_values[:-1] ** 2 / two_kappa
        v_return[idx] = factor * v.sum()
    return v_return


def dv_dt_piecewise(kappa: float,
                    vol: np.ndarray,
                    expiry_idx: int,
                    maturity_idx: int,
                    g_eg: np.ndarray,
                    event_grid: np.ndarray) -> np.ndarray:
    """Calculate 1st order time derivative of v-function.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        1st order time derivative of v-function.
    """
    return -vol[:expiry_idx + 1] ** 2 \
        * (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2 \
        * np.exp(2 * kappa * event_grid[:expiry_idx + 1])


def v_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step_ig: np.ndarray,
              vol_ig: np.ndarray,
              expiry_idx: int,
              maturity_idx: int,
              g_eg: np.ndarray) -> np.ndarray:
    """Calculate v-function on event grid until expiry.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.

    Returns:
        v-function.
    """
    factor = (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2
    int_kappa = np.cumsum(int_kappa_step_ig[:int_event_idx[expiry_idx] + 1])
    v_return = np.zeros(expiry_idx + 1)
    for event_idx in range(expiry_idx + 1):
        # Integration index of event.
        idx = int_event_idx[event_idx]
        # Integration index of expiry (+1).
        int_expiry_idx = int_event_idx[expiry_idx] + 1
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx:int_expiry_idx]
        # Slice of time-integrated kappa for each integration step.
        int_kappa_tmp = int_kappa[idx:int_expiry_idx]
        integrand = vol_ig[idx:int_expiry_idx] ** 2 * np.exp(2 * int_kappa_tmp)
        integrand *= factor
        v_return[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return v_return


def dv_dt_general(int_event_idx: np.ndarray,
                  int_kappa_step_ig: np.ndarray,
                  vol_ig: np.ndarray,
                  expiry_idx: int,
                  maturity_idx: int,
                  g_eg: np.ndarray) -> np.ndarray:
    """Calculate 1st order time derivative of v-function.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.

    Returns:
        1st order time derivative of v-function.
    """
    factor = (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2
    int_kappa = np.cumsum(int_kappa_step_ig[:int_event_idx[expiry_idx] + 1])
    v_return = np.zeros(expiry_idx + 1)
    for event_idx in range(expiry_idx + 1):
        # Integration index of event (+1).
        idx = int_event_idx[event_idx] + 1
        # Slice of time-integrated kappa for each integration step.
        int_kappa_tmp = int_kappa[:idx]
        v_return[event_idx] = \
            factor * vol_ig[idx - 1] ** 2 * np.exp(2 * int_kappa_tmp)
    return v_return


def option_price(spot: typing.Union[float, np.ndarray],
                 strike: float,
                 event_idx: int,
                 expiry_idx: int,
                 maturity_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option price.

    Price of European call or put option written on zero-coupon bond.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        option_type: European call or put option. Default is call.

    Returns:
        European call/put option price.
    """
    if option_type == Instrument.EUROPEAN_CALL:
        omega = 1
    elif option_type == Instrument.EUROPEAN_PUT:
        omega = -1
    else:
        raise ValueError(f"Option type is unknown: {option_type}")
    # P(t,T): Zero-coupon bond price at time zero with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)

    # v-function.
    v = v_eg[event_idx]

    # d-function.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    return omega * (price2 * norm.cdf(omega * d_plus)
                    - strike * price1 * norm.cdf(omega * d_minus))


def option_delta(spot: typing.Union[float, np.ndarray],
                 strike: float,
                 event_idx: int,
                 expiry_idx: int,
                 maturity_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option delta.

    Delta of European call or put option written on zero-coupon bond.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        option_type: European call or put option. Default is call.

    Returns:
        European call/put option delta.
    """
    if option_type == Instrument.EUROPEAN_CALL:
        omega = 1
    elif option_type == Instrument.EUROPEAN_PUT:
        omega = -1
    else:
        raise ValueError(f"Option type is unknown: {option_type}")
    # P(t,T): Zero-coupon bond price at time zero with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)

    # v-function.
    v = v_eg[event_idx]

    # d-function.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    # 1st order spatial derivative of d-function.
    d_delta = dd_dr(price1, delta1, price2, delta2, v)
    first_term = delta2 * norm.cdf(omega * d_plus) \
        - strike * delta1 * norm.cdf(omega * d_minus)
    second_term = price2 * norm.pdf(omega * d_plus) \
        - strike * price1 * norm.pdf(omega * d_minus)
    second_term *= d_delta
    return omega * first_term + omega ** 2 * second_term


def option_gamma(spot: typing.Union[float, np.ndarray],
                 strike: float,
                 event_idx: int,
                 expiry_idx: int,
                 maturity_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option gamma.

    Gamma of European call or put option written on zero-coupon bond.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        option_type: European call or put option. Default is call.

    Returns:
        European call/put option gamma.
    """
    if option_type == Instrument.EUROPEAN_CALL:
        omega = 1
    elif option_type == Instrument.EUROPEAN_PUT:
        omega = -1
    else:
        raise ValueError(f"Option type is unknown: {option_type}")
    # P(t,T): Zero-coupon bond price at time zero with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    gamma1 = zcbond.gamma(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)
    gamma2 = zcbond.gamma(spot, event_idx)

    # v-function.
    v = v_eg[event_idx]

    # d-function.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    # 1st order spatial derivative of d-function.
    d_delta = dd_dr(price1, delta1, price2, delta2, v)
    # 2nd order spatial derivative of d-function.
    d_gamma = d2d_dr2(price1, delta1, gamma1, price2, delta2, gamma2, v)
    first_term = \
        gamma2 * norm.cdf(omega * d_plus) \
        + delta2 * norm.pdf(omega * d_plus) * omega * d_delta \
        - gamma1 * strike * norm.cdf(omega * d_minus) \
        - delta1 * strike * norm.pdf(omega * d_minus) * omega * d_delta
    second_term = \
        delta2 * norm.pdf(omega * d_plus) * d_delta \
        + price2 * norm.pdf(omega * d_plus) \
        * (d_gamma - d_plus * d_delta ** 2) \
        - delta1 * strike * norm.pdf(omega * d_minus) * d_delta \
        - price1 * norm.pdf(omega * d_minus) \
        * (strike * d_gamma - d_minus * strike * d_delta ** 2)
    return omega * first_term + omega ** 2 * second_term


def option_theta(spot: typing.Union[float, np.ndarray],
                 strike: float,
                 event_idx: int,
                 expiry_idx: int,
                 maturity_idx: int,
                 zcbond,
                 v_eg: np.ndarray,
                 dv_dt_eg: np.ndarray,
                 option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option theta.

    Theta of European call or put option written on zero-coupon bond.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        dv_dt_eg: TODO...
        option_type: European call or put option. Default is call.

    Returns:
        European call/put option theta.
    """
    if option_type == Instrument.EUROPEAN_CALL:
        omega = 1
    elif option_type == Instrument.EUROPEAN_PUT:
        omega = -1
    else:
        raise ValueError(f"Option type is unknown: {option_type}")
    # P(t,T): Zero-coupon bond price at time zero with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    theta1 = zcbond.theta(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    theta2 = zcbond.theta(spot, event_idx)

    # v-function.
    v = v_eg[event_idx]
    dv_dt = dv_dt_eg[event_idx]

    # d-function.
    d_plus, d_minus = d_function(price1, price2, strike, v)

    # 1st order time derivative of d-function.
    d_theta = dd_dt(price1, theta1, price2, theta2, strike, v, dv_dt)

    first_term = theta2 * norm.cdf(omega * d_plus) \
        - strike * theta1 * norm.cdf(omega * d_minus)
    second_term = price2 * norm.pdf(omega * d_plus) * d_theta[0] \
        - strike * price1 * norm.pdf(omega * d_minus) * d_theta[1]
    return omega * first_term + omega ** 2 * second_term


def d_function(price1: typing.Union[float, np.ndarray],
               price2: typing.Union[float, np.ndarray],
               strike: float,
               v: float) -> tuple:
    """Calculate d-function.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time zero with maturity T.
        price2: Zero-coupon bond price at time zero with maturity T*.
        strike: Strike price of zero-coupon bond.
        v: Value of v-function at event.

    Returns:
        d-function.
    """
    d = np.log(price2 / (strike * price1))
    d_plus = (d + v / 2) / math.sqrt(v)
    d_minus = (d - v / 2) / math.sqrt(v)
    return d_plus, d_minus


def dd_dr(price1: typing.Union[float, np.ndarray],
          delta1: typing.Union[float, np.ndarray],
          price2: typing.Union[float, np.ndarray],
          delta2: typing.Union[float, np.ndarray],
          v: float) -> typing.Union[float, np.ndarray]:
    """Calculate 1st order spatial derivative of d-function.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time zero with maturity T.
        delta1: Delta of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time zero with maturity T*.
        delta2: Delta of zero-coupon bond price with maturity T*.
        v: Value of v-function at event.

    Returns:
        1st order spatial derivative of d-function.
    """
    return (delta2 / price2 - delta1 / price1) / math.sqrt(v)


def d2d_dr2(price1: typing.Union[float, np.ndarray],
            delta1: typing.Union[float, np.ndarray],
            gamma1: typing.Union[float, np.ndarray],
            price2: typing.Union[float, np.ndarray],
            delta2: typing.Union[float, np.ndarray],
            gamma2: typing.Union[float, np.ndarray],
            v: float) -> typing.Union[float, np.ndarray]:
    """Calculate 2nd order spatial derivative of d-function.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time zero with maturity T.
        delta1: Delta of zero-coupon bond price with maturity T.
        gamma1: Gamma of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time zero with maturity T*.
        delta2: Delta of zero-coupon bond price with maturity T*.
        gamma2: Gamma of zero-coupon bond price with maturity T*.
        v: Value of v-function at event.

    Returns:
        2nd order spatial derivative of d-function.
    """
    return (gamma2 / price2 - delta2 ** 2 / price2 ** 2
            - gamma1 / price1 + delta1 ** 2 / price1 ** 2) / math.sqrt(v)


def dd_dt(price1: typing.Union[float, np.ndarray],
          theta1: typing.Union[float, np.ndarray],
          price2: typing.Union[float, np.ndarray],
          theta2: typing.Union[float, np.ndarray],
          strike: float,
          v: float,
          dv_dt: float) -> tuple:
    """Calculate 1st order time derivative of d-function.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time zero with maturity T.
        theta1: Theta of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time zero with maturity T*.
        theta2: Theta of zero-coupon bond price with maturity T*.
        strike: Strike price of zero-coupon bond.
        v: Value of v-function at event.
        dv_dt: TODO...

    Returns:
        1st order time derivative of d-function.
    """
    d_plus, d_minus = d_function(price1, price2, strike, v)

#    dv_dt = 1

    factor = dv_dt / (2 * v)
    term = theta2 / price2 - theta1 / price1
    dd_plus_dt = -factor * d_plus + (term + dv_dt / 2) / math.sqrt(v)
    dd_minus_dt = -factor * d_minus + (term - dv_dt / 2) / math.sqrt(v)
    return dd_plus_dt, dd_minus_dt


def swap_schedule(fixing_start: int,
                  fixing_end: int,
                  fixing_frequency: int,
                  events_per_fixing: int) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """...

    Args:
        fixing_start: Year in which fixing starts.
        fixing_end: Year in which fixing ends.
        fixing_frequency: Yearly fixing frequency.
        events_per_fixing: Events per fixing period.

    Returns:
        ...
    """
    # Number of events from time zero to fixing_end, both included.
    n_events = fixing_end * fixing_frequency * events_per_fixing + 1
    # Time step between two adjacent events.
    dt = fixing_end / (n_events - 1)
    # Equidistant event grid.
    event_grid = dt * np.arange(n_events)
    # Number of fixings from fixing_start to fixing_end.
    n_fixings = (fixing_end - fixing_start) * fixing_frequency
    # Fixing and payment schedules.
    fixing_schedule = np.zeros(n_fixings, dtype=int)
    payment_schedule = np.zeros(n_fixings, dtype=int)
    # Index of first fixing.
    start_idx = fixing_start * fixing_frequency * events_per_fixing
    for n in range(n_fixings):
        fixing_schedule[n] = start_idx + n * events_per_fixing
        payment_schedule[n] = fixing_schedule[n] + events_per_fixing
    return event_grid, fixing_schedule, payment_schedule

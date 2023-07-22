import math
import typing

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

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


def y_constant(kappa: float,
               vol: float,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    Assuming that speed of mean reversion and volatility are constant.
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


def y_piecewise(kappa: float,
                vol: np.ndarray,
                event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    Assuming that speed of mean reversion is constant and volatility
    strip is piecewise constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
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


def y_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step_ig: np.ndarray,
              vol_ig: np.ndarray,
              event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate y-function on event and integration grid.

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


def g_function(maturity_idx: int,
               g_eg: np.ndarray,
               int_kappa_eg: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(t,t_maturity), on event grid.

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

    Assuming that speed of mean reversion is constant.
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


########################################################################


def int_y_constant(kappa: float,
                   vol: float,
                   event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    Assuming that speed of mean reversion and volatility are constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.

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


def double_int_y_constant(kappa: float,
                          vol: float,
                          event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    Assuming that speed of mean reversion and volatility are constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.

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


def int_y_piecewise(kappa: float,
                    vol: np.ndarray,
                    event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    Assuming that speed of mean reversion is constant and volatility
    strip is piecewise constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

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
        delta_t = event_grid[idx] - event_grid[idx - 1]
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


def double_int_y_piecewise(kappa: float,
                           vol: np.ndarray,
                           event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    Assuming that speed of mean reversion is constant and volatility
    strip is piecewise constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.42).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

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
        delta_t = event_grid[idx] - event_grid[idx - 1]
        y = math.exp(-kappa * delta_t)
        delta_t = event_grid[idx] + event_grid[idx - 1] - 2 * vol_times[-1]
        y += math.exp(-kappa * delta_t)
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


def int_y_general(int_grid: np.ndarray,
                  int_event_idx: np.ndarray,
                  int_kappa_step: np.ndarray,
                  vol_ig: np.ndarray,
                  event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        "Integral" of y-function.
    """
    y_eg, y_ig = y_general(int_grid, int_event_idx, int_kappa_step,
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
        int_kappa = np.append(int_kappa_step[idx1 + 1:idx2], 0)
        int_kappa = np.cumsum(int_kappa[::-1])[::-1]
        integrand = np.exp(-int_kappa) * y_ig[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return integral


def double_int_y_general(int_grid: np.ndarray,
                         int_event_idx: np.ndarray,
                         int_kappa_step: np.ndarray,
                         vol_ig: np.ndarray,
                         event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        "Double integral" of y-function.
    """
    y_eg, y_ig = y_general(int_grid, int_event_idx, int_kappa_step,
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
            int_kappa_tmp = np.append(int_kappa_step[idx1 + 1:idx + 1], 0)
            int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]
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

    Assuming that speed of mean reversion is constant and volatility
    strip is constant. The function doesn't include the instantaneous
    forward rate. See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        alpha-function.
    """
    return vol ** 2 * (1 - np.exp(-kappa * event_grid)) ** 2 / (2 * kappa ** 2)


def alpha_piecewise(kappa: float,
                    vol: np.ndarray,
                    event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    Assuming that speed of mean reversion is constant and volatility
    strip is piecewise constant. The function doesn't include the
    instantaneous forward rate. See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        alpha-function.
    """
    two_kappa_sq = 2 * kappa ** 2
    sum_array = np.zeros(event_grid.size)
    for idx in range(1, event_grid.size):
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        #
        delta_t = event_grid[idx] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[:-1]) - np.exp(-kappa * delta_t[1:])
        tmp *= vol_values[:-1] ** 2 / two_kappa_sq
        sum_array[idx] += tmp.sum()
        delta_t = event_grid[idx] - 2 * vol_times[-1]
        tmp = math.exp(kappa * event_grid[idx]) + math.exp(-kappa * delta_t)
        sum_array[idx] += vol_values[-1] ** 2 * tmp / two_kappa_sq
        #
        delta_t = event_grid[idx - 1] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[:-1]) - np.exp(-kappa * delta_t[1:])
        tmp *= vol_values[:-1] ** 2 / two_kappa_sq
        sum_array[idx] -= tmp.sum()
        delta_t = event_grid[idx - 1] - 2 * vol_times[-1]
        tmp = \
            math.exp(kappa * event_grid[idx - 1]) + math.exp(-kappa * delta_t)
        sum_array[idx] -= vol_values[-1] ** 2 * tmp / two_kappa_sq
    integral = np.zeros(event_grid.size)
    for idx in range(1, event_grid.size):
        factor = math.exp(-kappa * event_grid[idx])
        integral[idx] = factor * np.sum(sum_array[:idx + 1])
    return integral


def alpha_general(int_grid: np.ndarray,
                  int_event_idx: np.ndarray,
                  int_kappa_step: np.ndarray,
                  vol_ig: np.ndarray,
                  event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    The function doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        alpha-function.
    """
    y_eg, y_ig = y_general(int_grid, int_event_idx, int_kappa_step,
                           vol_ig, event_grid)
    integral = np.zeros(event_grid.size)
    for event_idx in range(1, event_grid.size):
        # Integration index of event.
        idx = int_event_idx[event_idx] + 1
        # Slice of integration grid.
        int_grid_tmp = int_grid[:idx]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = np.append(int_kappa_step[1:idx], 0)
        int_kappa = np.cumsum(int_kappa[::-1])[::-1]
        integrand = np.exp(-int_kappa) * y_ig[:idx]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return integral


def int_alpha_constant(kappa: float,
                       vol: float,
                       event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    Assuming that speed of mean reversion is constant and volatility
    strip is constant. The integrand doesn't include the instantaneous
    forward rate. See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

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


def int_alpha_piecewise(kappa: float,
                        vol: np.ndarray,
                        event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    Assuming that speed of mean reversion is constant and volatility
    strip is piecewise constant. The integrand doesn't include the
    instantaneous forward rate. See Pelsser, section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility strip on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        Integral of alpha-function.
    """
    two_kappa = 2 * kappa
    two_kappa_cubed = 2 * kappa ** 3
    sum_array = np.zeros(event_grid.size)
    for idx in range(1, event_grid.size):
        event_filter = event_grid < event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        #
        delta_t = event_grid[idx] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        tmp *= vol_values[:-1] ** 2 / two_kappa_cubed
        sum_array[idx] += tmp.sum()
        delta_t = event_grid[idx] - 2 * vol_times[-1]
        tmp = math.exp(kappa * event_grid[idx]) + math.exp(-kappa * delta_t)
        sum_array[idx] -= vol_values[-1] ** 2 * tmp / two_kappa_cubed
        #
        delta_t = event_grid[idx - 1] - 2 * vol_times
        tmp = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        tmp *= vol_values[:-1] ** 2 / two_kappa_cubed
        sum_array[idx] -= tmp.sum()
        delta_t = event_grid[idx - 1] - 2 * vol_times[-1]
        tmp = \
            math.exp(kappa * event_grid[idx - 1]) + math.exp(-kappa * delta_t)
        sum_array[idx] += vol_values[-1] ** 2 * tmp / two_kappa_cubed

    integral = np.zeros(event_grid.size)
    for i_index in range(1, event_grid.size):

        #
        event_filter = event_grid < event_grid[i_index]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]

        factor = math.exp(-kappa * event_grid[i_index]) \
            - math.exp(-kappa * event_grid[i_index - 1])
        integral[i_index] = np.sum(factor * sum_array[:i_index])

        #
        delta_t = event_grid[i_index] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / (2 * two_kappa_cubed)
        integral[i_index] += y.sum()
        delta_t = event_grid[i_index] - event_grid[i_index - 1]
        y = math.exp(-two_kappa * delta_t) - two_kappa * event_grid[i_index]
        y *= vol_values[-1] ** 2 / (2 * two_kappa_cubed)
        integral[i_index] -= y

        #
        delta_t = event_grid[i_index - 1] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / (2 * two_kappa_cubed)
        integral[i_index] -= y.sum()
        y = 1 - two_kappa * event_grid[i_index - 1]
        y *= vol_values[-1] ** 2 / (2 * two_kappa_cubed)
        integral[i_index] += y

        #
        delta_t = event_grid[i_index] + event_grid[i_index - 1] - 2 * vol_times
        y = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[i_index] -= y.sum()
        delta_t = event_grid[i_index] - event_grid[i_index - 1]
        y = math.exp(-kappa * delta_t)
        delta_t = event_grid[i_index] + event_grid[i_index - 1] - 2 * vol_times[-1]
        y += math.exp(-kappa * delta_t)
        y *= vol_values[-1] ** 2 / two_kappa_cubed
        integral[i_index] += y

        #
        delta_t = event_grid[i_index - 1] + event_grid[i_index - 1] - 2 * vol_times
        y = np.exp(-kappa * delta_t[1:]) - np.exp(-kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[i_index] += y.sum()
        delta_t = event_grid[i_index - 1] - event_grid[i_index - 1]
        y = math.exp(-kappa * delta_t)
        delta_t = event_grid[i_index - 1] + event_grid[i_index - 1] - 2 * vol_times[-1]
        y += math.exp(-kappa * delta_t)
        y *= vol_values[-1] ** 2 / two_kappa_cubed
        integral[i_index] -= y

    return integral


def int_alpha_general(int_grid: np.ndarray,
                      int_event_idx: np.ndarray,
                      int_kappa_step: np.ndarray,
                      vol_ig: np.ndarray,
                      event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser, section 5.3.

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        Integral of alpha-function.
    """
    y_eg, y_ig = y_general(int_grid, int_event_idx, int_kappa_step,
                           vol_ig, event_grid)
    integral = np.zeros(event_grid.size)
    for event_idx in range(1, event_grid.size):
        # Integration indices of two adjacent events.
        idx1 = int_event_idx[event_idx - 1]
        idx2 = int_event_idx[event_idx] + 1
        # Double time integral in Eq. (10.42).
        inner_integral = np.array(0)
        for idx in range(idx1 + 1, idx2):

            # int_grid_tmp = int_grid[idx1:idx + 1]
            int_grid_tmp = int_grid[:idx + 1]

            # int_kappa_tmp = np.append(int_kappa_step[idx1 + 1:idx + 1], 0)
            int_kappa_tmp = np.append(int_kappa_step[1:idx + 1], 0)

            int_kappa_tmp = np.cumsum(int_kappa_tmp[::-1])[::-1]

            # integrand = np.exp(-int_kappa_tmp) * y_ig[idx1:idx + 1]
            integrand = np.exp(-int_kappa_tmp) * y_ig[:idx + 1]

            inner_integral = \
                np.append(inner_integral,
                          np.sum(misc.trapz(int_grid_tmp, integrand)))
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, inner_integral))
    return integral


def v_constant(kappa: float,
               vol: float,
               expiry_idx: int,
               maturity_idx: int,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate v-function on event grid until expiry.

    Assuming that speed of mean reversion and volatility are constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        v-function.
    """
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    factor1 = (1 - math.exp(-kappa * (maturity - expiry))) ** 2
    factor2 = 1 - np.exp(-2 * kappa * (expiry - event_grid[:expiry_idx + 1]))
    return vol ** 2 * factor1 * factor2 / (2 * kappa ** 3)


def v_piecewise(kappa: float,
                vol: np.ndarray,
                expiry_idx: int,
                maturity_idx: int,
                event_grid: np.ndarray) -> np.ndarray:
    """Calculate v-function on event grid until expiry.

    Assuming that speed of mean reversion is constant and volatility is
    piecewise constant. See L.B.G. Andersen & V.V. Piterbarg 2010,
    proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        v-function.
    """
    two_kappa = 2 * kappa
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    factor = (math.exp(-kappa * expiry) - math.exp(-kappa * maturity)) ** 2
    factor /= kappa ** 2
    # Event grid until expiry.
    event_grid_expiry = event_grid[:expiry_idx + 1]
    # Vol strip until expiry.
    vol_expiry = vol[:expiry_idx + 1]
    v_return = np.zeros(expiry_idx + 1)
    for idx in range(expiry_idx):
        vol_times = event_grid_expiry[idx:]
        vol_values = vol_expiry[idx:]
        v = np.exp(two_kappa * vol_times[1:]) \
            - np.exp(two_kappa * vol_times[:-1])
        v *= vol_values[:-1] ** 2 / two_kappa
        v_return[idx] = factor * v.sum()
    return v_return


def call_put_price(spot: typing.Union[float, np.ndarray],
                   strike: float,
                   event_idx: int,
                   expiry_idx: int,
                   maturity_idx: int,
                   zcbond,
                   v_eg: np.ndarray,
                   type_: str) -> typing.Union[float, np.ndarray]:
    """Price function wrt value of pseudo short rate.

    Price of European call or put option written on zero-coupon bond.
    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Current value of pseudo short rate.
        strike: Strike value of underlying zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg:
        type_: Type of European option, call or put. Default is call.

    Returns:
        Price of call or put option.
    """
    # P(t,T): Zero-coupon bond price at time zero with maturity T.
    zcbond.maturity_idx = expiry_idx
    zcbond.initialization()
    price1 = zcbond.price(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
    zcbond.maturity_idx = maturity_idx
    zcbond.initialization()
    price2 = zcbond.price(spot, event_idx)
    # v-function.
    v = v_eg[event_idx]
    # d-function.
    d = np.log(price2 / (strike * price1))
    d_plus = (d + v / 2) / math.sqrt(v)
    d_minus = (d - v / 2) / math.sqrt(v)
    if type_ == "call":
        sign = 1
    elif type_ == "put":
        sign = -1
    else:
        raise ValueError(f"Option type unknown: {type_}")
    return sign * price2 * norm.cdf(sign * d_plus) \
        - sign * strike * price1 * norm.cdf(sign * d_minus)


def call_put_delta(spot: typing.Union[float, np.ndarray],
                   strike: float,
                   event_idx: int,
                   expiry_idx: int,
                   maturity_idx: int,
                   zcbond,
                   v_eg: np.ndarray,
                   type_: str) -> typing.Union[float, np.ndarray]:
    """1st order price sensitivity wrt value of pseudo short rate.

    Delta of European call or put option written on zero-coupon bond.
    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 4.5.1, and
    D. Brigo & F. Mercurio 2007, section 3.3.

    Args:
        spot: Current value of pseudo short rate.
        strike: Strike value of underlying zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg:
        type_: Type of European option, call or put. Default is call.

    Returns:
        Delta of call or put option.
    """
    # P(t,T): Zero-coupon bond price at time zero with maturity T.
    zcbond.maturity_idx = expiry_idx
    zcbond.initialization()
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time zero with maturity T*.
    zcbond.maturity_idx = maturity_idx
    zcbond.initialization()
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)
    # v-function.
    v = v_eg[event_idx]
    # d-function.
    d = np.log(price2 / (strike * price1))
    d_plus = (d + v / 2) / math.sqrt(v)
    d_minus = (d - v / 2) / math.sqrt(v)
    # Derivative of d-function.
    d_delta = (delta2 / price2 - delta1 / price1) / math.sqrt(v)
    if type_ == "call":
        sign = 1
    elif type_ == "put":
        sign = -1
    else:
        raise ValueError(f"Option type unknown: {type_}")
    first_terms = sign * delta2 * norm.cdf(sign * d_plus) \
        - sign * strike * delta1 * norm.cdf(sign * d_minus)
    last_terms = sign * price2 * norm.pdf(sign * d_plus) \
        - sign * strike * price1 * norm.pdf(sign * d_minus)
    last_terms *= sign * d_delta
    return first_terms + last_terms


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

import math

import numpy as np
from scipy.interpolate import UnivariateSpline

from utils import misc


def setup_model_parameters(inst):
    """Set up model parameters on event and integration grids.

    Args:
        inst: Financial instrument object.
    """
    # Kappa interpolated on event grid.
    inst.kappa_eg = inst.kappa.interpolation(inst.event_grid)
    # Vol interpolated on event grid.
    inst.vol_eg = inst.vol.interpolation(inst.event_grid)
    # Discount curve interpolated on event grid.
    inst.discount_curve_eg = inst.discount_curve.interpolation(inst.event_grid)
    # Instantaneous forward rate on event grid. TODO: Test accuracy of derivative!
    log_discount = np.log(inst.discount_curve_eg)
    smoothing = 0
    log_discount_spline = UnivariateSpline(
        inst.event_grid, log_discount, s=smoothing)
    forward_rate = log_discount_spline.derivative()
    inst.forward_rate_eg = -forward_rate(inst.event_grid)
    # Kappa and vol are constant.
    if inst.time_dependence == "constant":
        # Integration of kappa on event grid.
        inst.int_kappa_eg = inst.kappa_eg[0] * inst.event_grid
        # G-function, G(0,t), on event grid.
        inst.g_eg = g_constant(inst.kappa_eg[0], inst.event_grid)
        # y-function on event grid.
        inst.y_eg = y_constant(
            inst.kappa_eg[0], inst.vol_eg[0], inst.event_grid)
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
        inst.int_kappa_step_ig = (
            np.append(0, misc.trapz(inst.int_grid, inst.kappa_ig)))
        # Integration of kappa on event grid.
        inst.int_kappa_eg = np.zeros(inst.event_grid.size)
        for event_idx, int_idx in enumerate(inst.int_event_idx):
            inst.int_kappa_eg[event_idx] = \
                np.sum(inst.int_kappa_step_ig[:int_idx + 1])
        # G-function, G(0,t), on event grid.
        inst.g_eg = g_general(
            inst.int_grid, inst.int_event_idx, inst.int_kappa_step_ig,
            inst.event_grid)
        # y-function on event grid.
        inst.y_eg, _ = y_general(
            inst.int_grid, inst.int_event_idx, inst.int_kappa_step_ig,
            inst.vol_ig, inst.event_grid)
    else:
        raise ValueError(f"Unknown time dependence: {inst.time_dependence}")


def integration_grid(
        event_grid: np.ndarray,
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
        initial_time = event_grid[idx]
        if steps == 0:
            grid = np.array(initial_time + step_size)
        else:
            grid = int_dt * np.arange(1, steps + 1) + initial_time
            diff_step = step_size - steps * int_dt
            if diff_step > 1.0e-12:
                grid = np.append(grid, grid[-1] + diff_step)
        int_grid = np.append(int_grid, grid)
        int_event_idx = np.append(int_event_idx, grid.size)
    int_event_idx = np.cumsum(int_event_idx)
    return int_grid, int_event_idx


def g_function(
        maturity_idx: int,
        g_eg: np.ndarray,
        int_kappa_eg: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(t,t_maturity), on event grid.

    See Andersen & Piterbarg (2010), Remark 10.1.9.

    Args:
        maturity_idx: Maturity index on event grid.
        g_eg: G-function, G(0,t) on event grid.
        int_kappa_eg: Integral of kappa on event grid.

    Returns:
        G-function.
    """
    return (g_eg[maturity_idx] - g_eg) * np.exp(int_kappa_eg)


def g_constant(
        kappa: float,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(0,t), on event grid.

    The speed of mean reversion is constant.

    See Andersen & Piterbarg (2010), Proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        G-function.
    """
    return (1 - np.exp(-kappa * event_grid)) / kappa


def g_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate G-function, G(0,t), on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See Andersen & Piterbarg (2010), Proposition 10.1.7.

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


def y_constant(
        kappa: float,
        vol: float,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    The speed of mean reversion and volatility are constant.

    See Andersen & Piterbarg (2010), Proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        y-function.
    """
    two_kappa = 2 * kappa
    return vol ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa


def int_y_constant(
        kappa: float,
        vol: float,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    The speed of mean reversion and volatility are constant.

    See Andersen & Piterbarg (2010), Eq. (10.40).

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


def int_int_y_constant(
        kappa: float,
        vol: float,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    The speed of mean reversion and volatility are constant.

    See Andersen & Piterbarg (2010), Eq. (10.42).

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "Double integral" of y-function.
    """
    exp_kappa_1 = (np.exp(-2 * kappa * event_grid[:-1])
                   - np.exp(-2 * kappa * event_grid[1:])) / 2
    event_grid_diff = np.diff(event_grid)
    exp_kappa_2 = np.exp(-kappa * event_grid_diff) - 1
    event_grid_sum = event_grid[1:] + event_grid[:-1]
    exp_kappa_3 = \
        np.exp(-kappa * event_grid_sum) - np.exp(-2 * kappa * event_grid[:-1])
    integral = np.zeros(event_grid.size)
    integral[1:] = \
        vol ** 2 * (kappa * event_grid_diff + exp_kappa_1
                    + exp_kappa_2 + exp_kappa_3) / (2 * kappa ** 3)
    return integral


def y_piecewise(
        kappa: float,
        vol: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate y-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See Andersen & Piterbarg (2010), Proposition 10.1.7.

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


def int_y_piecewise(
        kappa: float,
        vol: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See Andersen & Piterbarg (2010), Eq. (10.40).

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
        y = (np.exp(-two_kappa * delta_t[:-1])
             - np.exp(-two_kappa * delta_t[1:]))
        y *= vol_values[:-1] ** 2 / two_kappa_sq
        integral[idx] += y.sum()
        # Second term.
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


def int_int_y_piecewise(
        kappa: float,
        vol: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See Andersen & Piterbarg (2010), Eq. (10.42).

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


def y_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate y-function on event and integration grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See Andersen & Piterbarg (2010), Proposition 10.1.7.

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


def int_y_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "integral" of y-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See Andersen & Piterbarg (2010), Eq. (10.40).

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
    _, y_ig = y_general(
        int_grid, int_event_idx, int_kappa_step_ig, vol_ig, event_grid)
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
        # Cumulative sum from "right to left".
        int_kappa = np.flip(np.cumsum(np.flip(int_kappa)))
        integrand = np.exp(-int_kappa) * y_ig[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return integral


def int_int_y_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "double integral" of y-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See Andersen & Piterbarg (2010), Eq. (10.42).

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
    _, y_ig = y_general(
        int_grid, int_event_idx, int_kappa_step_ig, vol_ig, event_grid)
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
            # Slice of time-integrated kappa for each integration step.
            int_kappa_tmp = np.append(int_kappa_step_ig[idx1 + 1:idx + 1], 0)
            # Cumulative sum from "right to left".
            int_kappa_tmp = np.flip(np.cumsum(np.flip(int_kappa_tmp)))
            integrand = np.exp(-int_kappa_tmp) * y_ig[idx1:idx + 1]
            inner_integral = np.append(
                inner_integral, np.sum(misc.trapz(int_grid_tmp, integrand)))
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx1:idx2]
        integral[event_idx] = np.sum(misc.trapz(int_grid_tmp, inner_integral))
    return integral


###############################################################################


def alpha_constant(
        kappa: float,
        vol: float,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    The speed of mean reversion and volatility are constant.

    The function doesn't include the instantaneous forward rate.
    See Pelsser (2000), Section 5.3.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        alpha-function.
    """
    return vol ** 2 * (1 - np.exp(-kappa * event_grid)) ** 2 / (2 * kappa ** 2)


def int_alpha_constant(
        kappa: float,
        vol: float,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    The speed of mean reversion and volatility are constant.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser (2000), Section 5.3.

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


def alpha_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    The function doesn't include the instantaneous forward rate.
    See Pelsser (2000), section 5.3.

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


def int_alpha_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser (2000), Section 5.3.

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


def alpha_piecewise(
        kappa: float,
        vol: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate alpha-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    The function doesn't include the instantaneous forward rate.
    See Pelsser (2000), Section 5.3.

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


def int_alpha_piecewise(
        kappa: float,
        vol: np.ndarray,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate integral of alpha-function on event grid.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    The integrand doesn't include the instantaneous forward rate.
    See Pelsser (2000), Section 5.3.

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

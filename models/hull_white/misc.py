import math
import typing

import numpy as np
from scipy.stats import norm

from models.hull_white import zero_coupon_bond
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

    Assuming that speed of mean reversion and volatility are constant.
    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

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
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        y-function.
    """
    y_function = np.zeros(event_grid.size)
    two_kappa = 2 * kappa
    for idx in range(1, event_grid.size):
        # See notes for "less than or equal to".
        event_filter = event_grid <= event_grid[idx]
        vol_times = event_grid[event_filter]
        vol_values = vol[event_filter]
        delta_t = event_grid[idx] - vol_times
        y = np.exp(-two_kappa * delta_t[1:]) \
            - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa
        y_function[idx] = y.sum()
    return y_function


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
        y = two_kappa * event_grid[idx] + math.exp(-two_kappa * delta_t)
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
        y = np.exp(-two_kappa * delta_t[1:]) - np.exp(-two_kappa * delta_t[:-1])
        y *= vol_values[:-1] ** 2 / two_kappa_cubed
        integral[idx] += y.sum()
        y = 2 * vol_values[-1] ** 2 / two_kappa_cubed
        integral[idx] -= y

    return integral


def y_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step: np.ndarray,
              vol_int_grid: np.ndarray,
              event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate y-function on event and integration grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

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


def int_y_general(int_grid: np.ndarray,
                  int_event_idx: np.ndarray,
                  int_kappa_step: np.ndarray,
                  vol_int_grid: np.ndarray,
                  event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate "integral" of y-function on event grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_int_grid: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        "Integral" of y-function.
    """
    y_eg, y_ig = y_general(int_grid, int_event_idx, int_kappa_step,
                           vol_int_grid, event_grid)
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
                         vol_int_grid: np.ndarray,
                         event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate "double integral" of y-function on event grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, Eq. (10.40).

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        vol_int_grid: Volatility on integration grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        "Double integral" of y-function.
    """
    y_eg, y_ig = y_general(int_grid, int_event_idx, int_kappa_step,
                           vol_int_grid, event_grid)
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


def g_constant(kappa: float,
               maturity_idx: int,
               event_grid: np.ndarray) -> np.ndarray:
    """Calculate G-function on event grid.

    Assuming that speed of mean reversion is constant. See
    L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        kappa: Speed of mean reversion.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        G-function.
    """
    maturity = event_grid[maturity_idx]
    # Event grid until maturity.
    grid = event_grid[event_grid <= maturity]
    return (1 - np.exp(-kappa * (maturity - grid))) / kappa


def g_general(int_grid: np.ndarray,
              int_event_idx: np.ndarray,
              int_kappa_step: np.ndarray,
              maturity_idx: int,
              event_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """Calculate G-function on event grid and integration grid.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.7.

    Args:
        int_grid: Integration grid.
        int_event_idx: Integration grid
        int_kappa_step: Step-wise integration of kappa on integration
            grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.

    Returns:
        G-function.
    """
    # Index of maturity on integration grid.
    int_mat_idx = int_event_idx[maturity_idx]
    # Calculation of G-function on integration grid.
    g_ig = np.zeros(int_mat_idx + 1)
    for idx in range(int_mat_idx + 1):
        # Slice of integration grid.
        int_grid_slice = int_grid[idx:int_mat_idx + 1]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = int_kappa_step[idx:int_mat_idx + 1]
        # Integrand in expression for G.
        integrand = np.exp(-np.cumsum(int_kappa))
        g_ig[idx] = np.sum(misc.trapz(int_grid_slice, integrand))
    # Save G-function on event grid.
    g_eg = np.zeros(event_grid.size)
    for event_idx, int_idx in enumerate(int_event_idx):
        g_eg[event_idx] = g_ig[int_idx]
    return g_eg, g_ig


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
                   zcbond: zero_coupon_bond.ZCBondNew,
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
                   zcbond: zero_coupon_bond.ZCBondNew,
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

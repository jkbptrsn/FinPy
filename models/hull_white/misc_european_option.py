import math
import typing

import numpy as np
from scipy.stats import norm

from utils.global_types import Instrument
from utils import misc


def v_function(
        expiry_idx: int,
        maturity_idx: int,
        g_eg: np.ndarray,
        v_eg: np.ndarray) -> np.ndarray:
    """Calculate v- or dv_dt-function on event grid until expiry.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
        g_eg: G-function, G(0,t), on event grid.
        v_eg: "v-function" on event grid until expiry.

    Returns:
        v- og dv_dt-function.
    """
    return (g_eg[maturity_idx] - g_eg[expiry_idx]) ** 2 * v_eg


def v_constant(
        kappa: float,
        vol: float,
        expiry_idx: int,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "v-function" on event grid until expiry.

    The speed of mean reversion is constant and volatility is constant.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Option expiry index on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "v-function".
    """
    two_kappa = 2 * kappa
    exp_kappa1 = math.exp(two_kappa * event_grid[expiry_idx])
    exp_kappa2 = np.exp(two_kappa * event_grid[:expiry_idx + 1])
    return vol ** 2 * (exp_kappa1 - exp_kappa2) / two_kappa


def dv_dt_constant(
        kappa: float,
        vol: float,
        expiry_idx: int,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate 1st order time derivative of "v-function".

    The speed of mean reversion is constant and volatility is constant.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Option expiry index on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        1st order time derivative of "v-function".
    """
    return -vol ** 2 * np.exp(2 * kappa * event_grid[:expiry_idx + 1])


def v_piecewise(
        kappa: float,
        vol: np.ndarray,
        expiry_idx: int,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate "v-function" on event grid until expiry.

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Option expiry index on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        "v-function".
    """
    two_kappa = 2 * kappa
    v_return = np.zeros(expiry_idx + 1)
    for idx in range(expiry_idx):
        vol_times = event_grid[idx:expiry_idx + 1]
        vol_values = vol[idx:expiry_idx + 1]
        v = np.exp(two_kappa * vol_times[1:]) \
            - np.exp(two_kappa * vol_times[:-1])
        v *= vol_values[:-1] ** 2 / two_kappa
        v_return[idx] = v.sum()
    return v_return


def dv_dt_piecewise(
        kappa: float,
        vol: np.ndarray,
        expiry_idx: int,
        event_grid: np.ndarray) -> np.ndarray:
    """Calculate 1st order time derivative of "v-function".

    The speed of mean reversion is constant and volatility is piecewise
    constant.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        kappa: Speed of mean reversion.
        vol: Volatility.
        expiry_idx: Option expiry index on event grid.
        event_grid: Event dates as year fractions from as-of date.

    Returns:
        1st order time derivative of "v-function".
    """
    return -vol[:expiry_idx + 1] ** 2 \
        * np.exp(2 * kappa * event_grid[:expiry_idx + 1])


###############################################################################


def v_general(
        int_grid: np.ndarray,
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        expiry_idx: int) -> np.ndarray:
    """Calculate "v-function" on event grid until expiry.

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        int_grid: Integration grid.
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        expiry_idx: Option expiry index on event grid.

    Returns:
        "v-function".
    """
    int_kappa = np.cumsum(int_kappa_step_ig[:int_event_idx[expiry_idx] + 1])
    v_return = np.zeros(expiry_idx + 1)
    for event_idx in range(expiry_idx):
        # Integration index of event.
        idx = int_event_idx[event_idx]
        # Integration index of expiry (+1).
        int_expiry_idx = int_event_idx[expiry_idx] + 1
        # Slice of integration grid.
        int_grid_tmp = int_grid[idx:int_expiry_idx]
        # Slice of time-integrated kappa for each integration step.
        int_kappa_tmp = int_kappa[idx:int_expiry_idx]
        integrand = vol_ig[idx:int_expiry_idx] ** 2 * np.exp(2 * int_kappa_tmp)
        v_return[event_idx] = np.sum(misc.trapz(int_grid_tmp, integrand))
    return v_return


def dv_dt_general(
        int_event_idx: np.ndarray,
        int_kappa_step_ig: np.ndarray,
        vol_ig: np.ndarray,
        expiry_idx: int) -> np.ndarray:
    """Calculate 1st order time derivative of "v-function".

    No assumption on the time dependence of the speed of mean reversion
    and the volatility.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        int_event_idx: Event indices on integration grid.
        int_kappa_step_ig: Step-wise integration of kappa on integration
            grid.
        vol_ig: Volatility on integration grid.
        expiry_idx: Option expiry index on event grid.

    Returns:
        1st order time derivative of "v-function".
    """
    int_kappa = np.cumsum(int_kappa_step_ig[:int_event_idx[expiry_idx] + 1])
    v_return = np.zeros(expiry_idx + 1)
    for event_idx in range(expiry_idx + 1):
        # Integration index of event.
        idx = int_event_idx[event_idx]
        v_return[event_idx] = -vol_ig[idx] ** 2 * np.exp(2 * int_kappa[idx])
    return v_return


def option_price(
        spot: typing.Union[float, np.ndarray],
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

    See Andersen & Piterbarg (2010), Proposition 4.5.1, and
    Brigo & Mercurio (2007), Section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
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
        raise ValueError(f"Unknown option type: {option_type}")
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time t with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    return omega * (price2 * norm.cdf(omega * d_plus)
                    - strike * price1 * norm.cdf(omega * d_minus))


def option_delta(
        spot: typing.Union[float, np.ndarray],
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

    See Andersen & Piterbarg (2010), Proposition 4.5.1, and
    Brigo & Mercurio (2007), Section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
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
        raise ValueError(f"Unknown option type: {option_type}")
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time t with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    # 1st order spatial derivative of d-function at time t.
    d_delta = dd_dr(price1, delta1, price2, delta2, v)
    first_term = delta2 * norm.cdf(omega * d_plus) \
        - strike * delta1 * norm.cdf(omega * d_minus)
    second_term = price2 * norm.pdf(omega * d_plus) \
        - strike * price1 * norm.pdf(omega * d_minus)
    second_term *= d_delta
    return omega * first_term + omega ** 2 * second_term


def option_gamma(
        spot: typing.Union[float, np.ndarray],
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

    See Andersen & Piterbarg (2010), Proposition 4.5.1, and
    Brigo & Mercurio (2007), Section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
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
        raise ValueError(f"Unknown option type: {option_type}")
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    delta1 = zcbond.delta(spot, event_idx)
    gamma1 = zcbond.gamma(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time t with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    delta2 = zcbond.delta(spot, event_idx)
    gamma2 = zcbond.gamma(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    # 1st order spatial derivative of d-function at time t.
    d_delta = dd_dr(price1, delta1, price2, delta2, v)
    # 2nd order spatial derivative of d-function at time t.
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


def option_theta(
        spot: typing.Union[float, np.ndarray],
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

    See Andersen & Piterbarg (2010), Proposition 4.5.1, and
    Brigo & Mercurio (2007), Section 3.3.

    Args:
        spot: Spot pseudo short rate.
        strike: Strike price of zero-coupon bond.
        event_idx: Index on event grid.
        expiry_idx: Option expiry index on event grid.
        maturity_idx: Bond maturity index on event grid.
        zcbond: Zero-coupon bond object.
        v_eg: v-function on event grid.
        dv_dt_eg: dv_dt-function on event grid.
        option_type: European call or put option. Default is call.

    Returns:
        European call/put option theta.
    """
    if option_type == Instrument.EUROPEAN_CALL:
        omega = 1
    elif option_type == Instrument.EUROPEAN_PUT:
        omega = -1
    else:
        raise ValueError(f"Unknown option type: {option_type}")
    # P(t,T): Zero-coupon bond price at time t with maturity T.
    zcbond.mat_idx = expiry_idx
    price1 = zcbond.price(spot, event_idx)
    theta1 = zcbond.theta(spot, event_idx)
    # P(t,T*): Zero-coupon bond price at time t with maturity T*.
    zcbond.mat_idx = maturity_idx
    price2 = zcbond.price(spot, event_idx)
    theta2 = zcbond.theta(spot, event_idx)
    # v-function at time t.
    v = v_eg[event_idx]
    # dv_dt-function at time t.
    dv_dt = dv_dt_eg[event_idx]
    # d-functions at time t.
    d_plus, d_minus = d_function(price1, price2, strike, v)
    # 1st order time derivative of d-functions at time t.
    d_theta = dd_dt(price1, theta1, price2, theta2, strike, v, dv_dt)
    first_term = theta2 * norm.cdf(omega * d_plus) \
        - strike * theta1 * norm.cdf(omega * d_minus)
    second_term = price2 * norm.pdf(omega * d_plus) * d_theta[0] \
        - strike * price1 * norm.pdf(omega * d_minus) * d_theta[1]
    return omega * first_term + omega ** 2 * second_term


def d_function(
        price1: typing.Union[float, np.ndarray],
        price2: typing.Union[float, np.ndarray],
        strike: float,
        v: float) -> tuple:
    """Calculate d-functions.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        strike: Strike price of zero-coupon bond.
        v: Value of v-function at time t.

    Returns:
        d-functions.
    """
    d = np.log(price2 / (strike * price1))
    d_plus = (d + v / 2) / math.sqrt(v)
    d_minus = (d - v / 2) / math.sqrt(v)
    return d_plus, d_minus


def dd_dr(
        price1: typing.Union[float, np.ndarray],
        delta1: typing.Union[float, np.ndarray],
        price2: typing.Union[float, np.ndarray],
        delta2: typing.Union[float, np.ndarray],
        v: float) -> typing.Union[float, np.ndarray]:
    """Calculate 1st order spatial derivative of d-functions.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        delta1: Delta of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        delta2: Delta of zero-coupon bond price with maturity T*.
        v: Value of v-function at time t.

    Returns:
        1st order spatial derivative of d-functions.
    """
    return (delta2 / price2 - delta1 / price1) / math.sqrt(v)


def d2d_dr2(
        price1: typing.Union[float, np.ndarray],
        delta1: typing.Union[float, np.ndarray],
        gamma1: typing.Union[float, np.ndarray],
        price2: typing.Union[float, np.ndarray],
        delta2: typing.Union[float, np.ndarray],
        gamma2: typing.Union[float, np.ndarray],
        v: float) -> typing.Union[float, np.ndarray]:
    """Calculate 2nd order spatial derivative of d-functions.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        delta1: Delta of zero-coupon bond price with maturity T.
        gamma1: Gamma of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        delta2: Delta of zero-coupon bond price with maturity T*.
        gamma2: Gamma of zero-coupon bond price with maturity T*.
        v: Value of v-function at time t.

    Returns:
        2nd order spatial derivative of d-functions.
    """
    return (gamma2 / price2 - delta2 ** 2 / price2 ** 2
            - gamma1 / price1 + delta1 ** 2 / price1 ** 2) / math.sqrt(v)


def dd_dt(
        price1: typing.Union[float, np.ndarray],
        theta1: typing.Union[float, np.ndarray],
        price2: typing.Union[float, np.ndarray],
        theta2: typing.Union[float, np.ndarray],
        strike: float,
        v: float,
        dv_dt: float) -> tuple:
    """Calculate 1st order time derivative of d-functions.

    See Andersen & Piterbarg (2010), Proposition 4.5.1.

    Args:
        price1: Zero-coupon bond price at time t with maturity T.
        theta1: Theta of zero-coupon bond price with maturity T.
        price2: Zero-coupon bond price at time t with maturity T*.
        theta2: Theta of zero-coupon bond price with maturity T*.
        strike: Strike price of zero-coupon bond.
        v: Value of v-function at time t.
        dv_dt: Value of dv_dt-function at time t.

    Returns:
        1st order time derivative of d-functions.
    """
    d_plus, d_minus = d_function(price1, price2, strike, v)
    factor = dv_dt / (2 * v)
    term = theta2 / price2 - theta1 / price1
    dd_plus_dt = -factor * d_plus + (term + dv_dt / 2) / math.sqrt(v)
    dd_minus_dt = -factor * d_minus + (term - dv_dt / 2) / math.sqrt(v)
    return dd_plus_dt, dd_minus_dt

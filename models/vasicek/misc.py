import math
import typing

import numpy as np
from scipy.stats import norm

from models.vasicek import zero_coupon_bond as zcbond
from utils.global_types import Instrument


def a_function(
        time1: float,
        time2: float,
        kappa: float,
        mean_rate: float,
        vol: float) -> float:
    """Calculate A-function.

    See Andersen & Piterbarg (2010), Proposition 10.1.4.

    Args:
        time1: Initial time.
        time2: Time of maturity.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.

    Returns:
        A-function.
    """
    vol_sq = vol ** 2
    two_kappa_sq = 2 * kappa ** 2
    b = b_function(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) \
        * (b - (time2 - time1)) - vol_sq * b ** 2 / (4 * kappa)


def b_function(
        time1: float,
        time2: float,
        kappa: float) -> float:
    """Calculate B-function.

    See Andersen & Piterbarg (2010), Proposition 10.1.4.

    Args:
        time1: Initial time.
        time2: Time of maturity.
        kappa: Speed of mean reversion.

    Returns:
        B-function.
    """
    return (1 - math.exp(-kappa * (time2 - time1))) / kappa


def dadt(
        time1: float,
        time2: float,
        kappa: float,
        mean_rate: float,
        vol: float) -> float:
    """Calculate 1st order partial derivative of A-function wrt time1.

    See Andersen & Piterbarg (2010), Proposition 10.1.4.

    Args:
        time1: Initial time.
        time2: Time of maturity.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.

    Returns:
        Time derivative of A-function.
    """
    vol_sq = vol ** 2
    two_kappa_sq = 2 * kappa ** 2
    b = b_function(time1, time2, kappa)
    db = dbdt(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) * (db + 1) \
        - vol_sq * b * db / (2 * kappa)


def dbdt(
        time1: float,
        time2: float,
        kappa: float) -> float:
    """Calculate 1st order partial derivative of B-function wrt time1.

    See Andersen & Piterbarg (2010), Proposition 10.1.4.

    Args:
        time1: Initial time.
        time2: Time of maturity.
        kappa: Speed of mean reversion.

    Returns:
        Time derivative of B-function.
    """
    return -math.exp(-kappa * (time2 - time1))


###############################################################################


def sigma_p(time1: float,
            time2: float,
            time3: float,
            kappa: float,
            vol: float) -> float:
    """Calculate sigma_p function.

    See Brigo & Mercurio (2007), Eq. (3.10).

    Args:
        time1: Initial time.
        time2: Time of option expiry.
        time3: Time of bond maturity.
        kappa: Speed of mean reversion.
        vol: Volatility.

    Returns:
        sigma_p function.
    """
    two_kappa = 2 * kappa
    exp_kappa = math.exp(-two_kappa * (time2 - time1))
    b = b_function(time2, time3, kappa)
    return vol * b * math.sqrt((1 - exp_kappa) / two_kappa)


def h_function(zc1_price: typing.Union[float, np.ndarray],
               zc2_price: typing.Union[float, np.ndarray],
               s_p: float,
               strike: float) -> typing.Union[float, np.ndarray]:
    """Calculate h function.

    See Brigo & Mercurio (2007), Eq. (3.10).

    Args:
        zc1_price: Price of zero-coupon bond with maturity at
            "option expiry".
        zc2_price: Price of zero-coupon bond with maturity at
            "bond maturity".
        s_p: sigma_p function.
        strike: Strike price of zero-coupon bond.

    Returns:
        h function.
    """
    return np.log(zc2_price / (zc1_price * strike)) / s_p + s_p / 2


def european_option_price(spot: typing.Union[float, np.ndarray],
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          event_grid: np.ndarray,
                          option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option price.

    See Brigo & Mercurio (2007), Eq. (3.10).

    Args:
        spot: Spot short rate.
        event_idx: Index on event grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        strike: Strike price of zero-coupon bond.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
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
    # Bond prices.
    zc1 = zcbond.ZCBond(kappa, mean_rate, vol, expiry_idx, event_grid)
    zc1_price = zc1.price(spot, event_idx)
    zc2 = zcbond.ZCBond(kappa, mean_rate, vol, maturity_idx, event_grid)
    zc2_price = zc2.price(spot, event_idx)
    # Event times.
    time = event_grid[event_idx]
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    # sigma_p and h functions.
    s_p = sigma_p(time, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    return omega * (zc2_price * norm.cdf(omega * h)
                    - strike * zc1_price * norm.cdf(omega * (h - s_p)))


def european_option_delta(spot: typing.Union[float, np.ndarray],
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          event_grid: np.ndarray,
                          option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option delta.

    See Brigo & Mercurio (2007), Eq. (3.10).

    Args:
        spot: Spot short rate.
        event_idx: Index on event grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        strike: Strike price of zero-coupon bond.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
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
    # Bond prices and deltas.
    zc1 = zcbond.ZCBond(kappa, mean_rate, vol, expiry_idx, event_grid)
    zc1_price = zc1.price(spot, event_idx)
    zc1_delta = zc1.delta(spot, event_idx)
    zc2 = zcbond.ZCBond(kappa, mean_rate, vol, maturity_idx, event_grid)
    zc2_price = zc2.price(spot, event_idx)
    zc2_delta = zc2.delta(spot, event_idx)
    # Event times.
    time = event_grid[event_idx]
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    # sigma_p and h functions.
    s_p = sigma_p(time, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    dhdr = (zc2_delta / zc2_price - zc1_delta / zc1_price) / s_p
    delta = zc2_delta * norm.cdf(omega * h) \
        - strike * zc1_delta * norm.cdf(omega * (h - s_p))
    delta += omega * dhdr \
        * (zc2_price * norm.pdf(omega * h)
           - strike * zc1_price * norm.pdf(omega * (h - s_p)))
    return omega * delta


def european_option_gamma(spot: typing.Union[float, np.ndarray],
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          event_grid: np.ndarray,
                          option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option gamma.

    See Brigo & Mercurio (2007), Eq. (3.10).

    Args:
        spot: Spot short rate.
        event_idx: Index on event grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        strike: Strike price of zero-coupon bond.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
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
    # Bond prices, deltas and gammas.
    zc1 = zcbond.ZCBond(kappa, mean_rate, vol, expiry_idx, event_grid)
    zc1_price = zc1.price(spot, event_idx)
    zc1_delta = zc1.delta(spot, event_idx)
    zc1_gamma = zc1.gamma(spot, event_idx)
    zc2 = zcbond.ZCBond(kappa, mean_rate, vol, maturity_idx, event_grid)
    zc2_price = zc2.price(spot, event_idx)
    zc2_delta = zc2.delta(spot, event_idx)
    zc2_gamma = zc2.gamma(spot, event_idx)
    # Event times.
    time = event_grid[event_idx]
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    # sigma_p and h functions.
    s_p = sigma_p(time, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    dhdr = (zc2_delta / zc2_price - zc1_delta / zc1_price) / s_p
    d2hdr2 = (- zc2_delta ** 2 / zc2_price ** 2 + zc2_gamma / zc2_price
              + zc1_delta ** 2 / zc1_price ** 2 - zc1_gamma / zc1_price) / s_p
    gamma = zc2_gamma * norm.cdf(omega * h) \
        - strike * zc1_gamma * norm.cdf(omega * (h - s_p))
    gamma += 2 * omega * dhdr * \
        (zc2_delta * norm.pdf(omega * h)
         - strike * zc1_delta * norm.pdf(omega * (h - s_p)))
    # See notes.
    gamma -= dhdr ** 2 * \
        (zc2_price * omega * h * norm.pdf(omega * h)
         - strike * zc1_price * omega * (h - s_p) * norm.pdf(omega * (h - s_p)))
    gamma += omega * d2hdr2 * \
        (zc2_price * norm.pdf(omega * h)
         - strike * zc1_price * norm.pdf(omega * (h - s_p)))
    return omega * gamma


def european_option_theta(spot: typing.Union[float, np.ndarray],
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          event_grid: np.ndarray,
                          option_type: Instrument = Instrument.EUROPEAN_CALL) \
        -> typing.Union[float, np.ndarray]:
    """Calculate European call/put option theta.

    See Brigo & Mercurio (2007), Eq. (3.10).

    Args:
        spot: Spot short rate.
        event_idx: Index on event grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        strike: Strike price of zero-coupon bond.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
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
    # Bond prices and deltas.
    zc1 = zcbond.ZCBond(kappa, mean_rate, vol, expiry_idx, event_grid)
    zc1_price = zc1.price(spot, event_idx)
    zc1_theta = zc1.theta(spot, event_idx)
    zc2 = zcbond.ZCBond(kappa, mean_rate, vol, maturity_idx, event_grid)
    zc2_price = zc2.price(spot, event_idx)
    zc2_theta = zc2.theta(spot, event_idx)
    # Event times.
    time = event_grid[event_idx]
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    # sigma_p and h functions.
    s_p = sigma_p(time, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    # 1st order time derivative of sigma_p and h functions.
    two_kappa = 2 * kappa
    exp_kappa = math.exp(-two_kappa * (expiry - time))
    ds_pdt = (math.sqrt((1 - exp_kappa) / two_kappa)) ** (-1)
    ds_pdt *= -vol * b_function(expiry, maturity, kappa) * exp_kappa / 2
    dhdt = (0.5 - np.log(zc2_price / (zc1_price * strike)) / s_p ** 2) * ds_pdt
    dhdt += (zc2_theta / zc2_price - zc1_theta / zc1_price) / s_p
    theta = zc2_theta * norm.cdf(omega * h) \
        - strike * zc1_theta * norm.cdf(omega * (h - s_p))
    theta += omega \
        * (zc2_price * norm.pdf(omega * h) * dhdt
           - strike * zc1_price * norm.pdf(omega * (h - s_p)) * (dhdt - ds_pdt))
    return omega * theta

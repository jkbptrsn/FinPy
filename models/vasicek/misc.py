import math
import numpy as np
from scipy.stats import norm

from models.vasicek import zero_coupon_bond


def a_function(time1: float,
               time2: float,
               kappa: float,
               mean_rate: float,
               vol: float) -> float:
    """Calculate A-function.

    See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.

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


def b_function(time1: float,
               time2: float,
               kappa: float) -> float:
    """Calculate B-function.

    See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.

    Args:
        time1: Initial time.
        time2: Time of maturity.
        kappa: Speed of mean reversion.

    Returns:
        B-function.
    """
    return (1 - math.exp(-kappa * (time2 - time1))) / kappa


def dadt(time1: float,
         time2: float,
         kappa: float,
         mean_rate: float,
         vol: float) -> float:
    """Calculate 1st order partial derivative of A-function wrt time.

    See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.

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
    db = dbdt(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) * (db + 1) \
        - vol_sq * b_function(time1, time2, kappa) * db / (2 * kappa)


def dbdt(time1: float,
         time2: float,
         kappa: float) -> float:
    """Calculate 1st order partial derivative of B-function wrt time.

    See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.

    Args:
        time1: Initial time.
        time2: Time of maturity.
        kappa: Speed of mean reversion.

    Returns:
        Time derivative of B-function.
    """
    return -math.exp(-kappa * (time2 - time1))


def sigma_p(time1: float,
            time2: float,
            time3: float,
            kappa: float,
            vol: float) -> float:
    """Calculate sigma_p-function.

    See Eq. (3.10), D. Brigo & F. Mercurio 2007.

    Args:
        time1: Initial time.
        time2: Time of expiry.
        time3: Time of maturity.
        kappa: Speed of mean reversion.
        vol: Volatility.

    Returns:
        sigma_p-function.
    """
    two_kappa = 2 * kappa
    exp_kappa = math.exp(-two_kappa * (time2 - time1))
    b = b_function(time2, time3, kappa)
    return vol * b * math.sqrt((1 - exp_kappa) / two_kappa)


def h_function(zc1_price: (float, np.ndarray),
               zc2_price: (float, np.ndarray),
               s_p: float,
               strike: float) -> (float, np.ndarray):
    """Calculate h-function.

    See Eq. (3.10), D. Brigo & F. Mercurio 2007.

    Args:
        zc1_price: Zero-coupon bond price at time of expiry.
        zc2_price: Zero-coupon bond price at time of maturity.
        s_p: sigma_p-function.
        strike: Strike price of zero-coupon bond at expiry.

    Returns:
        h-function.
    """
    return np.log(zc2_price / (zc1_price * strike)) / s_p + s_p / 2


def european_option_price(spot: float,
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          event_grid: np.ndarray,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          option_type: str = "Call"):
    """Calculate European call/put option price.

    See Eq. (3.10), D. Brigo & F. Mercurio 2007.

    Args:
        spot: Spot rate.
        event_idx: Index of current time on event_grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of zero-coupon bond at expiry.
        expiry_idx: Expiry index on event_grid.
        maturity_idx: Maturity index on event_grid.
        option_type: European call or put option.

    Returns:
        European call/put option price.
    """
    if option_type == "Call":
        omega = 1
    elif option_type == "Put":
        omega = -1
    else:
        raise Exception(f"Option type is unknown: {option_type}")
    zc1 = \
        zero_coupon_bond.ZCBondNew(kappa, mean_rate, vol, event_grid, expiry_idx)
    zc1_price = zc1.price(spot, event_idx)
    zc2 = zero_coupon_bond.ZCBondNew(kappa, mean_rate, vol, event_grid,
                                     maturity_idx)
    zc2_price = zc2.price(spot, event_idx)
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    s_p = sigma_p(event_idx, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    return omega * (zc2_price * norm.cdf(omega * h)
                    - strike * zc1_price * norm.cdf(omega * (h - s_p)))


def european_option_delta(spot: float,
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          event_grid: np.ndarray,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          option_type: str = "Call"):
    """Calculate European call/put option delta.

    See Eq. (3.10), D. Brigo & F. Mercurio 2007.

    Args:
        spot: Spot rate.
        event_idx: Index of current time on event_grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of zero-coupon bond at expiry.
        expiry_idx: Expiry index on event_grid.
        maturity_idx: Maturity index on event_grid.
        option_type: European call or put option.

    Returns:
        European call/put option delta.
    """
    if option_type == "Call":
        omega = 1
    elif option_type == "Put":
        omega = -1
    else:
        raise Exception(f"Option type is unknown: {option_type}")
    zc1 = \
        zero_coupon_bond.ZCBondNew(kappa, mean_rate, vol, event_grid, expiry_idx)
    zc1_price = zc1.price(spot, event_idx)
    zc1_delta = zc1.delta(spot, event_idx)
    zc2 = zero_coupon_bond.ZCBondNew(kappa, mean_rate, vol, event_grid,
                                     maturity_idx)
    zc2_price = zc2.price(spot, event_idx)
    zc2_delta = zc2.delta(spot, event_idx)
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    s_p = sigma_p(event_idx, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    dhdr = (zc2_delta / zc2_price - zc1_delta / zc1_price) / s_p
    delta = zc2_delta * norm.cdf(omega * h) \
        - strike * zc1_delta * norm.cdf(omega * (h - s_p))
    delta += omega * dhdr \
        * (zc2_price * norm.pdf(omega * h)
           - strike * zc1_price * norm.pdf(omega * (h - s_p)))
    return omega * delta


def european_option_gamma(spot: float,
                          event_idx: int,
                          kappa: float,
                          mean_rate: float,
                          vol: float,
                          event_grid: np.ndarray,
                          strike: float,
                          expiry_idx: int,
                          maturity_idx: int,
                          option_type: str = "Call"):
    """Calculate European call/put option gamma.

    See Eq. (3.10), D. Brigo & F. Mercurio 2007.

    Args:
        spot: Spot rate.
        event_idx: Index of current time on event_grid.
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of zero-coupon bond at expiry.
        expiry_idx: Expiry index on event_grid.
        maturity_idx: Maturity index on event_grid.
        option_type: European call or put option.

    Returns:
        European call/put option gamma.
    """
    if option_type == "Call":
        omega = 1
    elif option_type == "Put":
        omega = -1
    else:
        raise Exception(f"Option type is unknown: {option_type}")
    zc1 = \
        zero_coupon_bond.ZCBondNew(kappa, mean_rate, vol, event_grid, expiry_idx)
    zc1_price = zc1.price(spot, event_idx)
    zc1_delta = zc1.delta(spot, event_idx)
    zc1_gamma = zc1.gamma(spot, event_idx)
    zc2 = zero_coupon_bond.ZCBondNew(kappa, mean_rate, vol, event_grid,
                                     maturity_idx)
    zc2_price = zc2.price(spot, event_idx)
    zc2_delta = zc2.delta(spot, event_idx)
    zc2_gamma = zc2.gamma(spot, event_idx)
    expiry = event_grid[expiry_idx]
    maturity = event_grid[maturity_idx]
    s_p = sigma_p(event_idx, expiry, maturity, kappa, vol)
    h = h_function(zc1_price, zc2_price, s_p, strike)
    dhdr = (zc2_delta / zc2_price - zc1_delta / zc1_price) / s_p
    d2hdr2 = (- zc2_delta ** 2 / zc2_price ** 2 + zc2_gamma / zc2_price
              + zc1_delta ** 2 / zc1_price ** 2 - zc1_gamma / zc1_price) / s_p
    gamma = zc2_gamma * norm.cdf(omega * h) \
        - strike * zc1_gamma * norm.cdf(omega * (h - s_p))
    gamma += 2 * omega * dhdr * \
        (zc2_delta * norm.pdf(omega * h)
         - strike * zc1_delta * norm.pdf(omega * (h - s_p)))
    gamma -= (omega * dhdr) ** 2 * \
        (zc2_price * h * norm.pdf(omega * h)
         + strike * zc1_price * (h - s_p) * norm.pdf(omega * (h - s_p)))
    gamma += omega * d2hdr2 * \
        (zc2_price * norm.pdf(omega * h)
         - strike * zc1_price * norm.pdf(omega * (h - s_p)))
    return gamma

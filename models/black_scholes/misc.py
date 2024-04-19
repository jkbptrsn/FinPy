import math
import typing

import numpy as np


def d1d2(
        spot: typing.Union[float, np.ndarray],
        time: float,
        rate: float,
        vol: float,
        expiry: float,
        strike: float) \
        -> typing.Union[
            typing.Tuple[float, float],
            typing.Tuple[np.ndarray, np.ndarray]]:
    """'Functions' in Black-Scholes call option pricing formula.

    See Hull (2015), Chapter 15.

    Args:
        spot: Stock price at as-of date.
        time: Time of as-of date.
        rate: Interest rate.
        vol: Volatility.
        expiry: Time of expiry.
        strike: Strike price of stock at expiry.

    Returns:
        'Functions'.
    """
    d1 = np.log(spot / strike) + (rate + vol ** 2 / 2) * (expiry - time)
    d1 /= vol * math.sqrt(expiry - time)
    return d1, d1 - vol * math.sqrt(expiry - time)

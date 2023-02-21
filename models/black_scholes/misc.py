import typing

import numpy as np


def d1d2(spot: typing.Union[float, np.ndarray],
         time: float,
         rate,
         vol,
         expiry,
         strike,
         dividend) \
        -> typing.Union[
            typing.Tuple[float, float], typing.Tuple[np.ndarray, np.ndarray]]:
    """Factors in Black-Scholes formula.

    Args:
        spot: Current stock price.
        time: Current time.
        rate: Interest rate.
        vol: Volatility.
        expiry: Time of expiry.
        strike: Strike price of stock at expiry.
        dividend: Stock dividend.

    Returns:
        ...
    """
    spot *= np.exp(-dividend * (expiry - time))
    d1 = np.log(spot / strike) + (rate + vol ** 2 / 2) * (expiry - time)
    d1 /= vol * np.sqrt(expiry - time)
    return d1, d1 - vol * np.sqrt(expiry - time)

import typing

import numpy as np


def d1d2(spot: (float, np.ndarray),
         time: float,
         rate,
         vol,
         expiry,
         strike,
         dividend) \
        -> (typing.Tuple[float, float], typing.Tuple[np.ndarray, np.ndarray]):
    """Factors in Black-Scholes formula.
    - Returns Tuple[float, float] if spot is a float
    - Returns Tuple[np.ndarray, np.ndarray] otherwise
    """
    spot *= np.exp(-dividend * (expiry - time))
    d1 = np.log(spot / strike) + (rate + vol ** 2 / 2) * (expiry - time)
    d1 /= vol * np.sqrt(expiry - time)
    return d1, d1 - vol * np.sqrt(expiry - time)

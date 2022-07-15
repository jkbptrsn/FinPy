import math
import numpy as np


def dn(spot: (float, np.ndarray),
       time: float,
       expiry: float,
       strike: float,
       vol: float) -> (float, np.ndarray):
    """Factor in Bachelier formula."""
    # Time-to-maturity
    ttm = expiry - time
    return (spot - strike) / (vol * math.sqrt(ttm))

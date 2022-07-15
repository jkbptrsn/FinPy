import math
import numpy as np


def a_factor(time1: float,
             time2: float,
             kappa: float,
             mean_rate: float,
             vol: float) -> float:
    """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
    vol_sq = vol ** 2
    four_kappa = 4 * kappa
    two_kappa_sq = 2 * kappa ** 2
    b = b_factor(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) \
        * (b - (time2 - time1)) - vol_sq * b ** 2 / four_kappa


def b_factor(time1: float,
             time2: float,
             kappa: float) -> float:
    """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
    return (1 - math.exp(- kappa * (time2 - time1))) / kappa


def dadt(time1: float,
         time2: float,
         kappa: float,
         mean_rate: float,
         vol: float) -> float:
    """Time derivative of A
    Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
    """
    vol_sq = vol ** 2
    two_kappa = 2 * kappa
    two_kappa_sq = 2 * kappa ** 2
    db = dbdt(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) * (db + 1) \
        - vol_sq * b_factor(time1, time2, kappa) * db / two_kappa


def dbdt(time1: float,
         time2: float,
         kappa: float) -> float:
    """Time derivative of B
    Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
    """
    return -math.exp(- kappa * (time2 - time1))


def sigma_p(time1: float,
            time2: float,
            time3: float,
            kappa: float,
            vol: float) -> float:
    """Eq. (3.10), D. Brigo & F. Mercurio 2007."""
    two_kappa = 2 * kappa
    exp_kappa = math.exp(- two_kappa * (time2 - time1))
    b = b_factor(time2, time3, kappa)
    return vol * b * math.sqrt((1 - exp_kappa) / two_kappa)


def h_factor(zc1_price: (float, np.ndarray),
             zc2_price: (float, np.ndarray),
             s_p: float,
             strike: float) -> (float, np.ndarray):
    """Eq. (3.10), D. Brigo & F. Mercurio 2007."""
    return np.log(zc2_price / (zc1_price * strike)) / s_p + s_p / 2

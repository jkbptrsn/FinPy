import abc
import math
import numpy as np

import instruments.options as options
import models.vasicek.sde as sde
import models.vasicek.bonds as bonds


class VanillaOption(options.VanillaOption, sde.SDE):
    """Vanilla option in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 strike: float,
                 expiry: float):
        super().__init__(kappa, mean_rate, vol)
        self._strike = strike
        self._expiry = expiry

    @property
    @abc.abstractmethod
    def option_type(self):
        pass

    @property
    def strike(self) -> float:
        return self._strike

    @strike.setter
    def strike(self, strike_):
        self._strike = strike_

    @property
    def expiry(self) -> float:
        return self._expiry

    @expiry.setter
    def expiry(self, expiry_):
        self._expiry = expiry_


def sigma_p(time1: float,
            time2: float,
            time3: float,
            kappa: float,
            vol: float) -> float:
    """Eq. (3.10), Brigo & Mercurio 2007."""
    two_kappa = 2 * kappa
    exp_kappa_time = math.exp(- two_kappa * (time2 - time1))
    b_factor = bonds.b_factor(time2, time3, kappa)
    return vol * b_factor * math.sqrt((1 - exp_kappa_time) / two_kappa)


def h_factor(zc1_price: (float, np.ndarray),
             zc2_price: (float, np.ndarray),
             s_p: float,
             strike: float) -> (float, np.ndarray):
    """Eq. (3.10), Brigo & Mercurio 2007."""
    return np.log(zc2_price / (zc1_price * strike)) / s_p + s_p / 2

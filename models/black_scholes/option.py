import abc
import math
import numpy as np
from typing import Tuple

import models.black_scholes.sde as sde
import models.option as option

# todo: Exotic option call, e.g. compound options with two strikes?


class VanillaOption(option.VanillaOption, sde.SDE):
    """Vanilla option in Black-Scholes model."""

    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol)
        self._strike = strike
        self._expiry = expiry

    @property
    @abc.abstractmethod
    def option_type(self):
        pass

    @property
    def strike(self):
        return self._strike

    @strike.setter
    def strike(self, strike_):
        self._strike = strike_

    @property
    def expiry(self):
        return self._expiry

    @expiry.setter
    def expiry(self, expiry_):
        self._expiry = expiry_

    def d1d2(self,
             spot: (float, np.ndarray),
             time: float) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):

        # todo: change math.sqrt to np.sqrt such that time could be an array

        """Factors in Black-Scholes formula"""
        d1 = np.log(spot / self.strike) \
            + (self.rate + self.vol ** 2 / 2) * (self.expiry - time)
        d1 /= self.vol * math.sqrt(self.expiry - time)
        return d1, d1 - self.vol * math.sqrt(self.expiry - time)

import functools
import math
import numpy as np
import models.black_scholes.sde as sde


class Option(sde.SDE):
    """
    European option in Black-Scholes model
    """
    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol)
        self._strike = strike
        self._expiry = expiry

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

    def d1d2(self, spot, time):
        """
        Factors in Black-Scholes formula

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float

        Returns
        -------
        float / numpy.ndarray
        """
        d1 = np.log(spot / self.strike) \
            + (self.rate + self.vol ** 2 / 2) * (self.expiry - time)
        d1 /= self.vol * math.sqrt(self.expiry - time)
        return d1, d1 - self.vol * math.sqrt(self.expiry - time)

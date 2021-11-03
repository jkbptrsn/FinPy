import math
import numpy as np
import models.black_scholes.sde as sde

# todo: One option class for 1 strike?
# todo: One option class for 2 strikes?
# todo: One option class not specifying number of strikes?


class Option(sde.SDE):
    """European option in Black-Scholes model
    todo: What about two strike, e.g. compound options?
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

    def d1d2(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        """Factors in Black-Scholes formula"""
        d1 = np.log(spot / self.strike) \
            + (self.rate + self.vol ** 2 / 2) * (self.expiry - time)
        d1 /= self.vol * math.sqrt(self.expiry - time)
        return d1, d1 - self.vol * math.sqrt(self.expiry - time)

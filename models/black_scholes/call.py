import math
import numpy as np
from scipy.stats import norm

import models.black_scholes.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class Call(option.VanillaOption):
    """European call option in Black-Scholes model."""

    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)
        self._option_type = global_types.OptionType.EUROPEAN_CALL

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.cdf(d1) \
            - self.strike * norm.cdf(d2) \
            * math.exp(-self.rate * (self.expiry - time))

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        d1, d2 = self.d1d2(spot, time)
        return norm.cdf(d1)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        d1, d2 = self.d1d2(spot, time)
        return norm.pdf(d1) / (spot * self.vol * math.sqrt(self.expiry - time))

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.pdf(d1) * math.sqrt(self.expiry - time)

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        d1, d2 = self.d1d2(spot, time)
        return - spot * norm.pdf(d1) * self.vol \
            / (2 * math.sqrt(self.expiry - time)) - self.rate * self.strike \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

    def rho(self,
            spot: (float, np.ndarray),
            time: float) -> (float, np.ndarray):
        d1, d2 = self.d1d2(spot, time)
        return self.strike * (self.expiry - time) \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

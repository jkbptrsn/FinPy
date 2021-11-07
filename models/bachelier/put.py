import math
import numpy as np
from scipy.stats import norm

import models.bachelier.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class Put(option.VanillaOption):
    """European put option in Bachelier model."""

    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)
        self._option_type = global_types.OptionType.EUROPEAN_PUT

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.put(spot, self.strike)

    def payoff_1st_deriv(self,
                         spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_cash_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return factor1 * norm.cdf(factor1 / factor2) \
            + factor2 * norm.pdf(factor1 / factor2)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return -norm.cdf(factor1 / factor2)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return norm.pdf(factor1 / factor2) / factor2

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return - norm.pdf(factor1 / factor2) / (2 * factor2)

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return math.sqrt(self.expiry - time) * norm.pdf(factor1 / factor2)

import math
import numpy as np
from scipy.stats import norm

import models.black_scholes.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class Put(option.VanillaOption):
    """European put option in Black-Scholes model."""

    def __init__(self,
                 rate: float,
                 vol: float,
                 strike: float,
                 expiry: float,
                 dividend: float = 0):
        super().__init__(rate, vol, strike, expiry, dividend)
        self._option_type = global_types.InstrumentType.EUROPEAN_PUT

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    def payoff(self,
               state: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.put(state, self.strike)

    def payoff_dds(self,
                   state: (float, np.ndarray)) -> (float, np.ndarray):
        """1st order partial derivative of payoff function wrt the
        underlying state."""
        return - payoffs.binary_cash_put(state, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """Price function."""
        d1, d2 = self.d1d2(spot, time)
        spot *= np.exp(-self.dividend * (self._expiry - time))
        return - spot * norm.cdf(-d1) \
            + self.strike * norm.cdf(-d2) \
            * math.exp(-self.rate * (self.expiry - time))

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        d1, d2 = self.d1d2(spot, time)
        return np.exp(-self.dividend * (self._expiry - time)) \
            * (norm.cdf(d1) - 1)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2st order price sensitivity wrt the underlying state."""
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.dividend * (self.expiry - time)) * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def rho(self,
            spot: (float, np.ndarray),
            time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt rate."""
        d1, d2 = self.d1d2(spot, time)
        return - self.strike * (self.expiry - time) \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        d1, d2 = self.d1d2(spot, time)
        spot *= math.exp(-self.dividend * (self.expiry - time))
        return - spot * norm.pdf(d1) * self.vol \
            / (2 * math.sqrt(self.expiry - time)) \
            + self.rate * self.strike \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2) \
            - self.dividend * spot * norm.cdf(d1)

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt volatility."""
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.pdf(d1) * math.sqrt(self.expiry - time)

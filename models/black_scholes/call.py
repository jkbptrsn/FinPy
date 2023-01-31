import math
import numpy as np
from scipy.stats import norm

import models.options as options
import models.black_scholes.misc as misc
import models.black_scholes.sde as sde
import utils.global_types as global_types
import utils.payoffs as payoffs


class Call(sde.SDE, options.VanillaOption):
    """European call option in Black-Scholes model.

    European call option written on stock price.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
        dividend: Stock dividend.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.EUROPEAN_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               state: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.call(state, self.strike)

    def payoff_dds(self,
                   state: (float, np.ndarray)) -> (float, np.ndarray):
        """1st order partial derivative of payoff function wrt the
        underlying state."""
        return payoffs.binary_cash_call(state, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """Price function."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= np.exp(-self.dividend * (self.expiry - time))
        return spot * norm.cdf(d1) \
            - self.strike * norm.cdf(d2) \
            * math.exp(-self.rate * (self.expiry - time))

    def delta(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return np.exp(-self.dividend * (self.expiry - time)) * norm.cdf(d1)

    def gamma(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """2st order price sensitivity wrt the underlying state."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return math.exp(-self.dividend * (self.expiry - time)) * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def rho(self,
            spot: (float, np.ndarray),
            time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt rate."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return self.strike * (self.expiry - time) \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

    def theta(self,
              spot: (float, np.ndarray),
              time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= math.exp(-self.dividend * (self.expiry - time))
        return - spot * norm.pdf(d1) * self.vol \
            / (2 * math.sqrt(self.expiry - time)) \
            - self.rate * self.strike \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2) \
            + self.dividend * spot * norm.cdf(d1)

    def vega(self,
             spot: (float, np.ndarray),
             time_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt volatility."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        spot *= math.exp(-self.dividend * (self.expiry - time))
        return spot * norm.pdf(d1) * math.sqrt(self.expiry - time)

import math
import numpy as np
from scipy.stats import norm

import models.bachelier.misc as misc
import models.bachelier.sde as sde
import utils.global_types as global_types
import utils.payoffs as payoffs


class Put(sde.SDE):
    """European put option in Bachelier model.

    European put option written on stock price.

    Attributes:
        rate: Interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event_grid.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.Instrument.EUROPEAN_PUT

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.put(spot, self.strike)

    def payoff_dds(self,
                   spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return - payoffs.binary_cash_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        dn = misc.dn(spot, time, self.expiry, self.strike, self.vol)
        # Time-to-maturity
        ttm = self.expiry - time
        # Discount factor
        discount = math.exp(-self.rate * ttm)
        return discount \
            * ((self.strike - spot) * norm.cdf(-dn)
               + self.vol * math.sqrt(self.expiry - time) * norm.pdf(dn))

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        """..."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def rho(self,
            spot: (float, np.ndarray),
            time: float) -> (float, np.ndarray):
        """..."""
        pass

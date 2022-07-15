import math
import numpy as np
from scipy.stats import norm

import models.bachelier.misc as misc
import models.bachelier.sde as sde
import utils.global_types as global_types
import utils.payoffs as payoffs


class Call(sde.SDE):
    """European call option in Bachelier model."""

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.EUROPEAN_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.call(spot, self.strike)

    def payoff_dds(self,
                   spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_cash_call(spot, self.strike)

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
            * ((spot - self.strike) * norm.cdf(dn)
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

    def rho(self,
            spot: (float, np.ndarray),
            time: float) -> (float, np.ndarray):
        """..."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        """..."""
        pass

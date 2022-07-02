import math
import numpy as np
from scipy.stats import norm

import models.bachelier.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class Put(option.VanillaOption):
    """European put option in Bachelier model."""

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid, strike, expiry_idx)

        self._option_type = global_types.InstrumentType.EUROPEAN_PUT

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

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
        dn = self.dn(spot, time)
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

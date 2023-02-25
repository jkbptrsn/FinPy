import math

import numpy as np
from scipy.stats import norm

from models import options
from models.bachelier import misc
from models.bachelier import sde
from utils import global_types
from utils import payoffs


class CallNew(options.EuropeanOptionAnalytical):
    """European call option in Bachelier model.

    TODO: Rename...

    European call option written on stock price.

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
                 strike: float,
                 expiry_idx: int,
                 event_grid: np.ndarray):
        super().__init__()
        self.rate = rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid

        self.type = global_types.Instrument.EUROPEAN_CALL
        self.model = global_types.Model.BACHELIER

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

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            # TODO: Use dt in propagation, with non-equidistant event grid...
            # Will this work for both theta-method implementations?
            self.fd.set_propagator()
            self.fd.propagation()

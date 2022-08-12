import numpy as np

from models import options
from models.vasicek import misc
from models.vasicek import sde
from utils import global_types
from utils import payoffs


class Put(sde.SDE, options.VanillaOption):
    """European put option in the Vasicek model.

    European put option written on a zero-coupon bond.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        strike: Strike price of zero-coupon bond at expiry.
        expiry_idx: Expiry index on event_grid.
        maturity_idx: Maturity index on event_grid.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int):
        super().__init__(kappa, mean_rate, vol, event_grid)
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx

        self.option_type = global_types.InstrumentType.EUROPEAN_PUT

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        return misc.european_option_price(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.event_grid, self.strike, self.expiry_idx, self.maturity_idx,
            "Put")

    def delta(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        return misc.european_option_delta(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.event_grid, self.strike, self.expiry_idx, self.maturity_idx,
            "Put")

    def gamma(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        return misc.european_option_gamma(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.event_grid, self.strike, self.expiry_idx, self.maturity_idx,
            "Put")

    def theta(self,
              spot: (float, np.ndarray),
              event_idx: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass

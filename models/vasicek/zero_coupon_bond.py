import numpy as np

from models import bonds
from models.vasicek import misc
from models.vasicek import sde
from utils import global_types
from utils import payoffs


class ZCBond(sde.SDE, bonds.Bond):
    """Zero-coupon bond in the Vasicek model.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        maturity_idx : Maturity index on event_grid.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 maturity_idx: int):
        super().__init__(kappa, mean_rate, vol, event_grid)
        self.maturity_idx = maturity_idx

        self.bond_type = global_types.InstrumentType.ZERO_COUPON_BOND

    def __repr__(self):
        return f"{self.bond_type} bond object"

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def a_function(self,
                   event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.a_function(event_time, self.maturity, self.kappa,
                               self.mean_rate, self.vol)

    def b_function(self,
                   event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.b_function(event_time, self.maturity, self.kappa)

    def dadt(self,
             event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.dadt(event_time, self.maturity, self.kappa, self.mean_rate,
                         self.vol)

    def dbdt(self,
             event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.dbdt(event_time, self.maturity, self.kappa)

    def payoff(self,
               spot_rate: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.zero_coupon_bond(spot_rate)

    def price(self,
              spot_rate: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Zero-coupon bond price.

        See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return np.exp(self.a_function(event_idx)
                      - self.b_function(event_idx) * spot_rate)

    def delta(self,
              spot_rate: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state.

        See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return -self.b_function(event_idx) * self.price(spot_rate, event_idx)

    def gamma(self,
              spot_rate: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state.

        See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return \
            self.b_function(event_idx) ** 2 * self.price(spot_rate, event_idx)

    def theta(self,
              spot_rate: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt time.

        See proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return self.price(spot_rate, event_idx) \
            * (self.dadt(event_idx) - self.dbdt(event_idx) * spot_rate)

import typing

import numpy as np

from models import bonds
from models.vasicek import misc
from models.vasicek import sde
from utils import global_types
from utils import payoffs


class ZCBond(bonds.VanillaBondAnalytical1F):
    """Zero-coupon bond in Vasicek model.

    See L.B.G. Andersen & V.V. Piterbarg 2010, proposition 10.1.4.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        event_grid: Event dates represented as year fractions from as-of
            date.
        maturity_idx: Maturity index on event_grid.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 maturity_idx: int):
        super().__init__()
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.event_grid = event_grid
        self.maturity_idx = maturity_idx

        self.type = global_types.Instrument.ZERO_COUPON_BOND
        self.model = global_types.Model.VASICEK

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
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current short rate.

        Returns:
            Payoff.
        """
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return np.exp(self.a_function(event_idx)
                      - self.b_function(event_idx) * spot)

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        return -self.b_function(event_idx) * self.price(spot, event_idx)

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        return \
            self.b_function(event_idx) ** 2 * self.price(spot, event_idx)

    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        return self.price(spot, event_idx) \
            * (self.dadt(event_idx) - self.dbdt(event_idx) * spot)

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.set_propagator()
            self.fd.propagation(dt)

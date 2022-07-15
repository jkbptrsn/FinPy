import math
import numpy as np

import models.bonds as bonds
import models.vasicek.sde as sde
import utils.global_types as global_types
import utils.payoffs as payoffs


class ZCBond(sde.SDE, bonds.Bond):
    """Zero-coupon bond in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 maturity_idx: int,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, mean_rate, vol, event_grid, int_step_size)
        self.maturity_idx = maturity_idx

        self.bond_type = global_types.InstrumentType.ZERO_COUPON_BOND

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def a_factor(self,
                 time: float) -> float:
        """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
        vol_sq = self.vol ** 2
        four_kappa = 4 * self.kappa
        two_kappa_sq = 2 * self.kappa ** 2
        b = self.b_factor(time)
        return (self.mean_rate - vol_sq / two_kappa_sq) \
            * (b - (self.maturity - time)) - vol_sq * b ** 2 / four_kappa

    def b_factor(self,
                 time: float) -> float:
        """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
        return \
            (1 - math.exp(- self.kappa * (self.maturity - time))) / self.kappa

    def dadt(self,
             time: float) -> float:
        """Time derivative of A
        Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        vol_sq = self.vol ** 2
        two_kappa = 2 * self.kappa
        two_kappa_sq = 2 * self.kappa ** 2
        db = self.dbdt(time)
        return (self.mean_rate - vol_sq / two_kappa_sq) * (db + 1) \
            - vol_sq * self.b_factor(time) * db / two_kappa

    def dbdt(self,
             time: float) -> float:
        """Time derivative of B
        Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return -math.exp(-self.kappa * (self.maturity - time))

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """Price function
        Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return np.exp(self.a_factor(time) - self.b_factor(time) * spot)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        return - self.b_factor(time) * self.price(spot, time)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2st order price sensitivity wrt the underlying state."""
        return self.b_factor(time) ** 2 * self.price(spot, time)

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        return self.price(spot, time) \
            * (self.dadt(time) - self.dbdt(time) * spot)

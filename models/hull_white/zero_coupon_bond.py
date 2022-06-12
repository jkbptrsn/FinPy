import math
import numpy as np

import models.hull_white.bonds as bonds
import utils.global_types as global_types
import utils.misc as misc
import utils.payoffs as payoffs


class ZCBond(bonds.Bond):
    """Zero-coupon bond class in Hull-White model."""

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 forward_rate: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 maturity_idx: int):
        super().__init__(kappa, vol, forward_rate, event_grid, maturity_idx)
        self._bond_type = global_types.InstrumentType.ZERO_COUPON_BOND

    @property
    def bond_type(self) -> global_types.InstrumentType:
        return self._bond_type

    @staticmethod
    def payoff(spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price of zero coupon bond.
        - Assuming that speed of mean reversion is time-independent.
        - Assuming event_grid[event_idx] < event_grid[maturity].
        """
        price_time1 = math.exp(self.forward_rate_contrib[event_idx, 1])
        price_time2 = math.exp(self.forward_rate_contrib[self.maturity, 1])
        delta_t = self.event_grid[self.maturity] - self.event_grid[event_idx]
        kappa = self.kappa.values[0]
        g = (1 - math.exp(-kappa * delta_t)) / kappa
        y = self.y_event_grid[event_idx]
        return price_time2 * np.exp(-spot * g - y * g ** 2 / 2) / price_time1

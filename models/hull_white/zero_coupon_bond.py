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
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 maturity_idx: int):
        super().__init__(kappa, vol, discount_curve, event_grid, maturity_idx)
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
        - spot is the pseudo rate, i.e., x(t)
        - Assuming event_grid[event_idx] < event_grid[maturity]
        Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # P(0,t)
        price1 = self.discount_curve.values[event_idx]
        # P(0,T)
        price2 = self.discount_curve.values[self.maturity_idx]
        # Integration indices of two adjacent event dates
        int_idx1 = self.int_event_idx[event_idx]
        int_idx2 = self.int_event_idx[self.maturity_idx] + 1
        # Slice of integration grid
        int_grid = self.int_grid[int_idx1:int_idx2]

        # Slice of time-integrated kappa for each integration step
        self.setup_kappa_vol_y()
        int_kappa = self.int_kappa_step[int_idx1:int_idx2]
        self.kappa_int_grid = None
        self.vol_int_grid = None
        self.y_int_grid = None
        self.int_kappa_step = None

        # G-function
        # Eq. (10.18), L.B.G. Andersen & V.V. Piterbarg 2010
        int_kappa = np.cumsum(int_kappa)
        integrand = np.exp(-int_kappa)
        g = np.sum(misc.trapz(int_grid, integrand))
        # y-function
        y = self.y_event_grid[event_idx]

        return price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1

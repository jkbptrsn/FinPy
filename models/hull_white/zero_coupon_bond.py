import numpy as np

import models.bonds as bonds
import models.hull_white.sde as sde
import utils.global_types as global_types
import utils.misc as misc
import utils.payoffs as payoffs


class ZCBond(sde.SDE, bonds.Bond):
    """Zero-coupon bond class in Hull-White model."""

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 maturity_idx: int,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.discount_curve = discount_curve
        self.maturity_idx = maturity_idx

        # Run relevant setup methods of parent SDE object
        self.setup_int_grid()
        self.setup_kappa_vol_y()
        self.kappa_int_grid = None
        self.vol_int_grid = None
        self.y_int_grid = None

        self.bond_type = global_types.InstrumentType.ZERO_COUPON_BOND

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff of zero coupon bond."""
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price of zero coupon bond."""
        return self.calc_price(spot, event_idx, self.maturity_idx)

    def price_vector(self,
                     spot: (float, np.ndarray),
                     event_idx: int,
                     maturity_indices: np.ndarray) -> np.ndarray:
        """Price of zero coupon bond for each index in maturity_indices."""
        if isinstance(spot, np.ndarray):
            zcbond_prices = np.zeros((maturity_indices.size, spot.size))
        else:
            zcbond_prices = np.zeros((maturity_indices.size, 1))
        for idx, maturity_idx in enumerate(maturity_indices):
            zcbond_prices[idx] = self.calc_price(spot, event_idx, maturity_idx)
        return zcbond_prices

    def calc_price(self,
                   spot:  (float, np.ndarray),
                   event_idx: int,
                   maturity_idx: int) ->  (float, np.ndarray):
        """Calculate price of zero coupon bond based on event_idx
        (time t) and maturity_idx (time T).
        - spot is the pseudo rate, i.e., x(t)
        - Assuming event_grid[event_idx] < event_grid[maturity_idx]
        Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # P(0,t)
        price1 = self.discount_curve.values[event_idx]
        # P(0,T)
        price2 = self.discount_curve.values[maturity_idx]
        # Integration indices of the two relevant event dates
        int_idx1 = self.int_event_idx[event_idx]
        int_idx2 = self.int_event_idx[maturity_idx] + 1
        # Slice of integration grid
        int_grid = self.int_grid[int_idx1:int_idx2]
        # Slice of time-integrated kappa for each integration step
        int_kappa = self.int_kappa_step[int_idx1:int_idx2]
        # G-function
        # Eq. (10.18), L.B.G. Andersen & V.V. Piterbarg 2010
        int_kappa = np.cumsum(int_kappa)
        integrand = np.exp(-int_kappa)
        g = np.sum(misc.trapz(int_grid, integrand))
        # y-function
        y = self.y_event_grid[event_idx]
        return price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1

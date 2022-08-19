import numpy as np

from models import bonds
from models.hull_white import sde
from utils import global_types
from utils import misc
from utils import payoffs


class ZCBond(sde.SDE, bonds.Bond):
    """Zero-coupon bond class for the 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.

        discount_curve: Discount curve represented on event-grid.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.

        maturity_idx: Event-grid index corresponding to maturity.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

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

        self.bond_type = global_types.InstrumentType.ZERO_COUPON_BOND

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        return self._calc_price(spot, event_idx, self.maturity_idx)

###############################################################################

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
            zcbond_prices[idx] = self._calc_price(spot, event_idx, maturity_idx)
        return zcbond_prices

###############################################################################

    def _calc_price(self,
                    spot: (float, np.ndarray),
                    event_idx: int,
                    maturity_idx: int) -> (float, np.ndarray):
        """Calculate zero-coupon bond price.

        Calculate price of zero-coupon bond based at current time
        (event_idx) for maturity at time T (maturity_idx). See
        proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.

        Assuming event_grid[event_idx] < event_grid[maturity_idx].

        Args:
            spot: Spot value of the pseudo short rate.
            event_idx: Event-grid index corresponding to current time.
            maturity_idx: Event-grid index corresponding to maturity.

        Returns:
            Zero-coupon bond price.
        """
        # P(0,t): Zero-coupon bond price at time zero with maturity t.
        price1 = self.discount_curve.values[event_idx]
        # P(0,T): Zero-coupon bond price at time zero with maturity T.
        price2 = self.discount_curve.values[maturity_idx]
        # Integration indices of the two relevant events.
        int_idx1 = self.int_event_idx[event_idx]
        int_idx2 = self.int_event_idx[maturity_idx] + 1
        # Slice of integration grid.
        int_grid = self.int_grid[int_idx1:int_idx2]
        # Slice of time-integrated kappa for each integration step.
        int_kappa = self.int_kappa_step[int_idx1:int_idx2]
        # G(t,T): G-function,
        # see Eq. (10.18), L.B.G. Andersen & V.V. Piterbarg 2010.
        integrand = np.exp(-np.cumsum(int_kappa))
        g = np.sum(misc.trapz(int_grid, integrand))
        # y(t): y-function,
        # see Eq. (10.17), L.B.G. Andersen & V.V. Piterbarg 2010.
        y = self.y_event_grid[event_idx]
        return price2 * np.exp(-spot * g - y * g ** 2 / 2) / price1

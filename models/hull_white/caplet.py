import math
import numpy as np
from scipy.stats import norm

from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from utils import global_types
from utils import misc


class Caplet(sde.SDE):
    """Caplet for the 1-factor Hull-White model.

    Note: The speed of mean reversion is assumed to be constant!

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Payment dates represented as year fractions from the
            as-of date.
        expiry_idx: Event grid index corresponding to expiry.
        maturity_idx: Event grid index corresponding to maturity.
        fixed_rate: Fixed rate.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 expiry_idx: int,
                 maturity_idx: int,
                 fixed_rate: float,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.fixed_rate = fixed_rate

        self.instrument_type = global_types.Instrument.CAPLET

        # Zero-coupon bond.
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    maturity_idx, int_step_size=int_step_size)

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See proposition 4.5.2, L.B.G. Andersen & V.V. Piterbarg 2010.

        Args:
            spot: Spot value of PSEUDO short rate.
            event_idx: Event-grid index of current time.

        Returns:
            Caplet price.
        """
        # Price of zero-coupon bond maturing at expiry_idx.
        self.zcbond.maturity_idx = self.expiry_idx
        bond_price_expiry = self.zcbond.price(spot, event_idx)
        # Price of zero-coupon bond maturing at maturity_idx.
        self.zcbond.maturity_idx = self.maturity_idx
        bond_price_maturity = self.zcbond.price(spot, event_idx)
        # Event-grid index corresponding to current time.
        int_event_idx1 = self.int_event_idx[event_idx]
        # Event-grid index corresponding to expiry.
        int_event_idx2 = self.int_event_idx[self.expiry_idx] + 1
        # Slice of integration grid.
        int_grid = self.int_grid[int_event_idx1:int_event_idx2]
        # Volatility strip on slice of integration grid.
        vol = self.vol_int_grid[int_event_idx1:int_event_idx2]
        # Constant kappa value.
        kappa = self.kappa.values[0]
        # v-function.
        integrand = vol ** 2 * np.exp(2 * kappa * int_grid)
        exp_kappa1 = math.exp(-kappa * self.event_grid[self.expiry_idx])
        exp_kappa2 = math.exp(-kappa * self.event_grid[self.maturity_idx])
        v = (exp_kappa1 - exp_kappa2) ** 2 \
            * np.sum(misc.trapz(int_grid, integrand)) / kappa ** 2
        # d-function.
        tau = self.maturity - self.expiry
        d = math.log((1 + self.fixed_rate * tau)
                     * bond_price_maturity / bond_price_expiry)
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)
        return bond_price_expiry * norm.cdf(-d_minus) \
            - ((1 + self.fixed_rate * tau) * bond_price_maturity
               * norm.cdf(-d_plus))

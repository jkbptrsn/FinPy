import numpy as np
from scipy.optimize import brentq

from models import options
from models.hull_white import put_option
from models.hull_white import sde
from models.hull_white import swap
from models.hull_white import zero_coupon_bond
from utils import global_types
from utils import misc


class Payer(sde.SDE, options.VanillaOption):
    """European payer swaption class for the 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Events, e.g. payment dates, represented as year
            fractions from the as-of date.
        expiry_idx: Event-grid index corresponding to expiry.
        maturity_idx: Event-grid index corresponding to maturity.
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
#        self.discount_curve = discount_curve
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.fixed_rate = fixed_rate

        self.option_type = global_types.Instrument.SWAPTION

        # Underlying swap
        # -- assuming swap maturity is equal to event_grid[-1]
        self.swap = swap.Swap(kappa, vol, discount_curve, event_grid,
                              fixed_rate, int_step_size=int_step_size)

        # Zero-coupon bond object with maturity at last event.
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    event_grid[-1],
                                    int_step_size=int_step_size)

        # Put option written on zero-coupon bond.
        self.put = put_option.Put(kappa, vol, discount_curve, event_grid,
                                  1, expiry_idx, maturity_idx,
                                  int_step_size=int_step_size)

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        pass

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See Eq. (10.24), L.B.G. Andersen & V.V. Piterbarg 2010.

        Args:
            spot: Spot value of PSEUDO short rate.
            event_idx: Event-grid index of current time.

        Returns:
            Swaption price.
        """
        swaption_price = 0
        # Pseudo short rate corresponding to zero swap value.
        rate_star = brentq(self.swap.price, -0.9, 0.9, args=(event_idx,))
        for mat_idx in range(event_idx + 1, self.event_grid.size):
            # "Strike" of put option
            self.zcbond.maturity_idx = mat_idx
            self.put.strike = self.zcbond.price(rate_star, event_idx)
            # Maturity of put option
            self.put.maturity_idx = mat_idx
            put_price = self.put.price(spot, event_idx)
            # Time between two adjacent payments in year fractions.
            tau = self.event_grid[mat_idx] - self.event_grid[mat_idx - 1]
            # ...
            swaption_price += self.fixed_rate * tau * put_price
            if mat_idx == self.event_grid.size - 1:
                swaption_price += put_price
        return swaption_price

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass

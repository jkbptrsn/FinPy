import numpy as np

from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from utils import global_types
from utils import misc


class Swap(sde.SDE):
    """Fixed-for-floating swap for the 1-factor Hull-White model.

    Attributes:
        kappa: Speed of mean reversion.
        vol: Volatility.
        discount_curve: Discount curve represented on event-grid.
        event_grid: Payment dates represented as year fractions from the
            as-of date.
        fixed_rate: Fixed rate.
        int_step_size: Integration/propagation step size represented as
            a year fraction. Default is 1 / 365.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 fixed_rate: float,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.fixed_rate = fixed_rate

        self.instrument_type = global_types.InstrumentType.SWAP

        # Zero-coupon bond object with maturity at last event.
        self.zcbond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                    event_grid[-1],
                                    int_step_size=int_step_size)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        See section 5.3, L.B.G. Andersen & V.V. Piterbarg 2010.

        Returns:
            Swap price.
        """
        swap_price = 0
        # Remaining event grid.
        event_grid_tmp = self.event_grid[event_idx:]
        for idx, tau in enumerate(np.diff(event_grid_tmp)):
            # Price of zero-coupon bond maturing at idx.
            self.zcbond.maturity_idx = event_idx + idx
            bond_price = self.zcbond.price(spot, event_idx)
            swap_price += bond_price
            # Price of zero-coupon bond maturing at idx + 1.
            self.zcbond.maturity_idx = event_idx + idx + 1
            bond_price = self.zcbond.price(spot, event_idx)
            swap_price -= (1 + tau * self.fixed_rate) * bond_price
        return swap_price

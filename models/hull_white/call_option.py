import math
import numpy as np
from scipy.stats import norm

import models.options as options
import models.hull_white.sde as sde
import models.hull_white.zero_coupon_bond as zcbond
import utils.global_types as global_types
import utils.misc as misc
import utils.payoffs as payoffs


class Call(sde.SDE, options.VanillaOption):
    """European call option written on zero-coupon bond in Hull-White
    model.

    Note: Assuming that speed of mean reversion is constant.
    """

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 discount_curve: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 int_step_size: float = 1 / 365):
        super().__init__(kappa, vol, event_grid, int_step_size)
        self.discount_curve = discount_curve
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx

        self.option_type = global_types.InstrumentType.EUROPEAN_CALL

        self.zcbond = \
            zcbond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.
        Proposition 4.5.1, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        self.zcbond.maturity_idx = self.expiry_idx
        price1 = self.zcbond.price(spot, event_idx)
        self.zcbond.maturity_idx = self.maturity_idx
        price2 = self.zcbond.price(spot, event_idx)

        self.setup_int_grid()
        int_event_idx1 = self.int_event_idx[event_idx]
        int_event_idx2 = self.int_event_idx[self.expiry_idx]
        int_grid = self.int_grid[int_event_idx1:int_event_idx2 + 1]
        vol = self.vol.interpolation(int_grid)

        kappa = self.kappa.values[0]

        integrand = vol ** 2 * np.exp(2 * kappa * int_grid)

        exp_kappa1 = math.exp(-kappa * self.event_grid[self.expiry_idx])
        exp_kappa2 = math.exp(-kappa * self.event_grid[self.maturity_idx])

        v = (exp_kappa1 - exp_kappa2) ** 2 \
            * np.sum(misc.trapz(int_grid, integrand)) / kappa ** 2

        d = math.log(price2 / (self.strike * price1))
        d_plus = (d + v / 2) / math.sqrt(v)
        d_minus = (d - v / 2) / math.sqrt(v)

        return price2 * norm.cdf(d_plus) \
            - self.strike * price1 * norm.cdf(d_minus)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2st order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass

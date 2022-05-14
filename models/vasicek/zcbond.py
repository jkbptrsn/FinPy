import math
import numpy as np

import models.vasicek.bonds as bond
import utils.global_types as global_types
import utils.payoffs as payoffs


class ZCBond(bond.Bond):
    """Zero-coupon bond in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 maturity: float):
        super().__init__(kappa, mean_rate, vol, maturity)
        self._option_type = global_types.InstrumentType.ZERO_COUPON_BOND

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.zero_coupon_bond(spot)

    def b_factor(self,
                 time: float) -> float:
        """Eq. (3.8), Brigo & Mercurio 2007."""
        return (1 - math.exp(- self.kappa * (self.maturity - time))) \
            / self.kappa

    def a_factor(self,
                 time: float) -> float:
        """Eq. (3.8), Brigo & Mercurio 2007."""
        vol_sq = self.vol ** 2
        four_kappa = 4 * self._kappa
        two_kappa_sq = 2 * self._kappa ** 2
        b = self.b_factor(time)
        return (self.mean_rate - vol_sq / two_kappa_sq) \
            * (b - self.maturity + time) - vol_sq * b ** 2 / four_kappa

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """Price function: Eq. (3.8), Brigo & Mercurio 2007."""
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

    def dbdt(self,
             time: float) -> float:
        """Time derivative of B: Eq. (3.8), Brigo & Mercurio 2007."""
        return - math.exp(- self.kappa * (self.maturity - time))

    def dadt(self,
             time: float) -> float:
        """Time derivative of A: Eq. (3.8), Brigo & Mercurio 2007."""
        vol_sq = self.vol ** 2
        two_kappa = 2 * self._kappa
        two_kappa_sq = 2 * self._kappa ** 2
        db = self.dbdt(time)
        return (self.mean_rate - vol_sq / two_kappa_sq) * (db + 1) \
            - vol_sq * self.b_factor(time) * db / two_kappa

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        return self.price(spot, time) \
            * (self.dadt(time) - self.dbdt(time) * spot)

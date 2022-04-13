import numpy as np

import models.vasicek.option as option
import utils.global_types as global_types


class ZCBond(option.VanillaOption):
    """Zero-coupon bond in Vasicek model."""

    def __init__(self, kappa, mean_rate, vol, strike, expiry):
        super().__init__(kappa, mean_rate, vol, strike, expiry)
        self._option_type = global_types.OptionType.ZERO_COUPON_BOND

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return 1 + 0 * spot

    def b_term(self, time: float) -> float:
        return (1 - np.exp(- self.kappa * (self.expiry - time))) / self.kappa

    def a_term(self, time: float) -> float:
        vol_sq = self.vol ** 2
        b_term = self.b_term(time)
        return (self.mean_rate - vol_sq / (2 * self.kappa ** 2)) \
            * (b_term - (self.expiry - time)) \
            - vol_sq * b_term ** 2 / (4 * self.kappa)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        return np.exp(self.a_term(time) - self.b_term(time) * spot)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        return - self.b_term(time) * self.price(spot, time)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        return self.b_term(time) ** 2 * self.price(spot, time)

import numpy as np
from scipy.stats import norm

import models.vasicek.options as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class Call(option.VanillaOption):
    """European call option on zero-coupon bond in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 strike: float,
                 expiry: float,
                 maturity: float):
        super().__init__(kappa, mean_rate, vol, strike, expiry)
        self._maturity = maturity
        self._option_type = global_types.InstrumentType.EUROPEAN_CALL

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    @property
    def maturity(self) -> float:
        return self._maturity

    @maturity.setter
    def maturity(self, maturity_):
        self._maturity = maturity_

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.call(spot, self.strike)

    def payoff_dds(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_cash_call(spot, self.strike)

    def b_term(self, time: float, maturity: float) -> float:
        # Change parameters to time1 and time2...
        return (1 - np.exp(- self.kappa * (maturity - time))) / self.kappa

    def a_term(self, time: float, maturity: float) -> float:
        # Change parameters to time1 and time2...
        vol_sq = self.vol ** 2
        b_term = self.b_term(time, maturity)
        return (self.mean_rate - vol_sq / (2 * self.kappa ** 2)) \
            * (b_term - (maturity - time)) \
            - vol_sq * b_term ** 2 / (4 * self.kappa)

    def zcbond(self,
               spot: float,
               time: float,
               maturity: float):
        return np.exp(self.a_term(time, maturity)
                      - self.b_term(time, maturity) * spot)

    def zcbond_ddr(self,
                   spot: float,
                   time: float,
                   maturity: float):
        return - self.b_term(time, maturity) \
            * self.zcbond(spot, time, maturity)

    def sigma_p(self,
                time: float):
        return self.vol \
            * np.sqrt((1 - np.exp(-2 * self.kappa * (self.expiry - time)))
                      / (2 * self.kappa)) \
            * self.b_term(self.expiry, self._maturity)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        zcbond_T = self.zcbond(spot, time, self.expiry)
        zcbond_S = self.zcbond(spot, time, self._maturity)
        sigma_p = self.sigma_p(time)
        h = np.log(zcbond_S / (zcbond_T * self.strike)) / sigma_p + sigma_p / 2
        return zcbond_S * norm.cdf(h) - self.strike * zcbond_T * norm.cdf(h - sigma_p)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        zcbond_T = self.zcbond(spot, time, self.expiry)
        zcbond_T_ddr = self.zcbond_ddr(spot, time, self.expiry)
        zcbond_S = self.zcbond(spot, time, self._maturity)
        zcbond_S_ddr = self.zcbond_ddr(spot, time, self._maturity)
        sigma_p = self.sigma_p(time)
        h = np.log(zcbond_S / (zcbond_T * self.strike)) / sigma_p + sigma_p / 2
        dhdr = (zcbond_S_ddr / zcbond_S - zcbond_T_ddr / zcbond_T) / sigma_p
        return zcbond_S_ddr * norm.cdf(h) - self.strike * zcbond_T_ddr * norm.cdf(h - sigma_p) + \
            dhdr * (zcbond_S * norm.pdf(h) - self.strike * zcbond_T * norm.pdf(h - sigma_p))

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):

        pass

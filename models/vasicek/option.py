import abc

import models.vasicek.sde as sde
import models.option as option


class VanillaOption(option.VanillaOption, sde.SDE):
    """Vanilla option in Vasicek model."""

    def __init__(self, kappa, mean_rate, vol, strike, expiry):
        super().__init__(kappa, mean_rate, vol)
        self._strike = strike
        self._expiry = expiry

    @property
    @abc.abstractmethod
    def option_type(self):
        pass

    @property
    def strike(self):
        return self._strike

    @strike.setter
    def strike(self, strike_):
        self._strike = strike_

    @property
    def expiry(self):
        return self._expiry

    @expiry.setter
    def expiry(self, expiry_):
        self._expiry = expiry_

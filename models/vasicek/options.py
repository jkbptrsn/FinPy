import abc

import instruments.options as options
import models.vasicek.sde as sde


class VanillaOption(options.VanillaOption, sde.SDE):
    """Vanilla option in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 strike: float,
                 expiry: float):
        super().__init__(kappa, mean_rate, vol)
        self._strike = strike
        self._expiry = expiry

    @property
    @abc.abstractmethod
    def option_type(self):
        pass

    @property
    def strike(self) -> float:
        return self._strike

    @strike.setter
    def strike(self, strike_):
        self._strike = strike_

    @property
    def expiry(self) -> float:
        return self._expiry

    @expiry.setter
    def expiry(self, expiry_):
        self._expiry = expiry_

import abc

import models.bachelier.sde as sde
import models.option as option


class VanillaOption(option.VanillaOption, sde.SDE):
    """Vanilla option in Bachelier model."""

    def __init__(self, vol, strike, expiry):
        super().__init__(vol)
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
    def strike(self, val):
        self._strike = val

    @property
    def expiry(self):
        return self._expiry

    @expiry.setter
    def expiry(self, val):
        self._expiry = val

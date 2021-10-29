import models.bachelier.sde as sde


class Option(sde.SDE):
    """
    European option in Bachelier model
    """
    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol)
        self._strike = strike
        self._expiry = expiry

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

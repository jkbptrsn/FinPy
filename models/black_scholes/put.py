import math
import numpy as np
from scipy.stats import norm
import models.black_scholes.option as option


class Put(option.Option):
    """
    European put option in Black-Scholes model
    """
    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)

    def payoff(self, spot):
        return np.maximum(self.strike - spot, 0)

    def price(self, spot, time):
        """
        Price of European put option

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float

        Returns
        -------
        float / numpy.ndarray
        """
        d1, d2 = self.d1d2(spot, time)
        return - spot * norm.cdf(-d1) \
            + self.strike * norm.cdf(-d2) \
            * math.exp(-self.rate * (self.expiry - time))

    def delta(self, spot, time):
        """Delta of European put option"""
        d1, d2 = self.d1d2(spot, time)
        return norm.cdf(d1) - 1

    def gamma(self, spot, time):
        """Gamma of European put option"""
        d1, d2 = self.d1d2(spot, time)
        return norm.pdf(d1) / (spot * self.vol * math.sqrt(self.expiry - time))

    def vega(self, spot, time):
        """Vega of European put option"""
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.pdf(d1) * math.sqrt(self.expiry - time)

    def theta(self, spot, time):
        """Theta of European put option"""
        d1, d2 = self.d1d2(spot, time)
        return - spot * norm.pdf(d1) * self.vol \
            / (2 * math.sqrt(self.expiry - time)) \
            + self.rate * self.strike \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def rho(self, spot, time):
        """Rho of European put option"""
        d1, d2 = self.d1d2(spot, time)
        return - self.strike * (self.expiry - time) \
            * math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

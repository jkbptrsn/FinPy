import math
from scipy.stats import norm

import models.bachelier.option as option
import utils.payoffs as payoffs


class Call(option.Option):
    """
    European call option in Bachelier model
    """
    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)

    def payoff(self, spot):
        """
        Payoff of European call option

        Parameters
        ----------
        spot : float / numpy.ndarray

        Returns
        -------
        float / numpy.ndarray
        """
        return payoffs.call(spot, self.strike)

    def price(self, spot, time):
        """
        Price of European call option in Bachelier model

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float

        Returns
        -------
        float / numpy.ndarray
        """
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return factor1 * norm.cdf(factor1 / factor2) \
            + factor2 * norm.pdf(factor1 / factor2)

    def delta(self, spot, time):
        """
        Delta of European call option in Bachelier model

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float

        Returns
        -------
        float / numpy.ndarray
        """
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return norm.cdf(factor1 / factor2)

    def gamma(self, spot, time):
        pass

    def vega(self, spot, time):
        pass

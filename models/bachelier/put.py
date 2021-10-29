import math
from scipy.stats import norm

import models.bachelier.option as option
import models.payoffs as payoffs


class Put(option.Option):
    """
    European put option in Bachelier model
    """
    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)

    def payoff(self, spot):
        """
        Payoff of European put option

        Parameters
        ----------
        spot : float / numpy.ndarray

        Returns
        -------
        float / numpy.ndarray
        """
        return payoffs.put(spot, self.strike)

    def price(self, spot, time):
        """
        Price of European put option in Bachelier model

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float

        Returns
        -------
        float / numpy.ndarray
        """
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return factor1 * norm.cdf(factor1 / factor2) \
            + factor2 * norm.pdf(factor1 / factor2)

    def delta(self, spot, time):
        """
        Delta of European put option in Bachelier model

        Parameters
        ----------
        spot : float / numpy.ndarray
        time : float

        Returns
        -------
        float / numpy.ndarray
        """
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return norm.cdf(factor1 / factor2)

    def gamma(self, spot, time):
        pass

    def vega(self, spot, time):
        pass

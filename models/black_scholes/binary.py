import math
import numpy as np
from scipy.special import erf
from scipy.stats import norm
import models.black_scholes.option as option


class Binary(option.Option):
    """
    European binary option in Black-Scholes model
    """
    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)

    def con_call_price(self, spot, time):
        """Price of European cash-or-nothing call option
        - Pays out one unit of cash if the spot is above the strike at
        expiry
        """
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

    def con_call_delta(self, spot, time):
        """Delta of European cash-or-nothing call option"""
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def aon_call_price(self, spot, time):
        """Price of European asset-or-nothing call option
        - Pays out one unit of the asset if the spot is above the strike at
        expiry
        """
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.cdf(d1)

    def aon_call_delta(self, spot, time):
        """Delta of European asset-or-nothing call option"""
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(d1)

    def con_put_price(self, spot, time):
        """Price of European cash-or-nothing put option
        - Pays out one unit of cash if the spot is below the strike at
        expiry
        """
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def con_put_delta(self, spot, time):
        """Delta of European cash-or-nothing put option"""
        d1, d2 = self.d1d2(spot, time)
        return - math.exp(-self.rate * (self.expiry - time)) * norm.pdf(-d2) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def aon_put_price(self, spot, time):
        """Price of European asset-or-nothing put option
        - Pays out one unit of the asset if the spot is below the strike at
        expiry
        """
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.cdf(-d1)

    def aon_put_delta(self, spot, time):
        """Delta of European asset-or-nothing put option"""
        d1, d2 = self.d1d2(spot, time)
        return - spot * norm.pdf(-d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(-d1)

    def con_american_price(self, spot, time):
        """Price of American cash-or-nothing call option"""
        a = np.log(self.strike / spot) / self.vol
        xi = self.rate / self.vol
        b = math.sqrt(xi ** 2 + 2 * self.rate)
        factor = np.exp(a * (xi - b)) / 2
        term_1 = np.sign(a) * erf((b * (self.expiry - time) - a)
                                  / math.sqrt(2 * (self.expiry - time)))
        term_2 = np.sign(a) * erf((b * (self.expiry - time) + a)
                                  / math.sqrt(2 * (self.expiry - time)))
        return factor * (1 + term_1 + np.exp(2 * a * b) * (1 - term_2))

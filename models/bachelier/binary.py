import math
from scipy.stats import norm

import models.bachelier.option as option
import utils.payoffs as payoffs


class BinaryCashCall(option.Option):
    """
    European cash-or-nothing call option: Pays out one unit of cash if
    the spot is above the strike at expiry
    """
    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)

    def payoff(self, spot):
        """
        Payoff of European cash-or-nothing call option

        Parameters
        ----------
        spot : float / numpy.ndarray

        Returns
        -------
        float / numpy.ndarray
        """
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self, spot, time):
        """
        Price of European cash-or-nothing call option in Bachelier model

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

    def delta(self, spot, time):
        """
        Delta of European cash-or-nothing call option in Bachelier model

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
        return norm.pdf(factor1 / factor2) / factor2

    def gamma(self, spot, time):
        pass

    def vega(self, spot, time):
        pass


class BinaryAssetCall(option.Option):
    """
    European asset-or-nothing call option: Pays out one unit of the
    asset if the spot is above the strike at expiry
    """
    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)

    def payoff(self, spot):
        """
        Payoff of European asset-or-nothing call option

        Parameters
        ----------
        spot : float / numpy.ndarray

        Returns
        -------
        float / numpy.ndarray
        """
        return payoffs.binary_asset_call(spot, self.strike)

    def price(self, spot, time):
        pass

    def delta(self, spot, time):
        pass

    def gamma(self, spot, time):
        pass

    def vega(self, spot, time):
        pass


class BinaryCashPut(option.Option):
    """
    European cash-or-nothing put option: Pays out one unit of cash if
    the spot is above the strike at expiry
    """
    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)

    def payoff(self, spot):
        """
        Payoff of European cash-or-nothing put option

        Parameters
        ----------
        spot : float / numpy.ndarray

        Returns
        -------
        float / numpy.ndarray
        """
        return payoffs.binary_cash_put(spot, self.strike)

    def price(self, spot, time):
        """
        Price of European cash-or-nothing put option in Bachelier model

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

    def delta(self, spot, time):
        """
        Delta of European cash-or-nothing put option in Bachelier model

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
        return - norm.pdf(factor1 / factor2) / factor2

    def gamma(self, spot, time):
        pass

    def vega(self, spot, time):
        pass


class BinaryAssetPut(option.Option):
    """
    European asset-or-nothing put option: Pays out one unit of the asset
    if the spot is above the strike at expiry
    """
    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)

    def payoff(self, spot):
        """
        Payoff of European asset-or-nothing put option

        Parameters
        ----------
        spot : float / numpy.ndarray

        Returns
        -------
        float / numpy.ndarray
        """
        return payoffs.binary_asset_put(spot, self.strike)

    def price(self, spot, time):
        pass

    def delta(self, spot, time):
        pass

    def gamma(self, spot, time):
        pass

    def vega(self, spot, time):
        pass

import math
import numpy as np
from scipy.stats import norm

import models.black_scholes.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class BinaryCashCall(option.VanillaOption):
    """European cash-or-nothing call option in Black-Scholes model.
    Pays out one unit of cash if the spot is above the strike at
    expiry."""

    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_CASH_CALL

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

    def delta(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2) \
            / (spot * self.vol * math.sqrt(self.expiry - time))


class BinaryAssetCall(option.VanillaOption):
    """European asset-or-nothing call option in Black-Scholes model.
    Pays out one unit of the asset if the spot is above the strike at
    expiry."""

    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_ASSET_CALL

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_asset_call(spot, self.strike)

    def price(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.cdf(d1)

    def delta(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(d1)


class BinaryCashPut(option.VanillaOption):
    """European cash-or-nothing put option in Black-Scholes model. Pays
    out one unit of cash if the spot is below the strike at expiry."""

    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_CASH_PUT

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_cash_put(spot, self.strike)

    def price(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def delta(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return - math.exp(-self.rate * (self.expiry - time)) * norm.pdf(-d2) \
            / (spot * self.vol * math.sqrt(self.expiry - time))


class BinaryAssetPut(option.VanillaOption):
    """European asset-or-nothing put option in Black-Scholes model. Pays
    out one unit of the asset if the spot is below the strike at
    expiry."""

    def __init__(self, rate, vol, strike, expiry):
        super().__init__(rate, vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_ASSET_PUT

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_asset_put(spot, self.strike)

    def price(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return spot * norm.cdf(-d1)

    def delta(self, spot, time):
        d1, d2 = self.d1d2(spot, time)
        return - spot * norm.pdf(-d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(-d1)

import math
import numpy as np
from scipy.stats import norm

import models.bachelier.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class BinaryCashCall(option.VanillaOption):
    """European cash-or-nothing call option in Bachelier model. Pays out
    one unit of cash if the spot is above the strike at expiry."""

    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_CASH_CALL

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return norm.cdf(factor1 / factor2)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return norm.pdf(factor1 / factor2) / factor2

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return - factor1 * self.delta(spot, time) / factor2

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return - (factor1 / factor2) ** 2 * norm.pdf(factor1 / factor2) / \
            (2 * (self.expiry - time))

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        factor1 = spot - self.strike
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return (factor1 / factor2) ** 2 * norm.pdf(factor1 / factor2)


class BinaryAssetCall(option.VanillaOption):
    """European asset-or-nothing call option in Bachelier model. Pays
    out one unit of the asset if the spot is above the strike at
    expiry."""

    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_ASSET_CALL

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_asset_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        pass


class BinaryCashPut(option.VanillaOption):
    """European cash-or-nothing put option in Bachelier model. Pays out
    one unit of cash if the spot is below the strike at expiry."""

    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_CASH_PUT

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot: (float, np.ndarray)) -> (float, np.ndarray):
        return payoffs.binary_cash_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return -norm.cdf(factor1 / factor2)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        factor1 = self.strike - spot
        factor2 = self.vol * math.sqrt(self.expiry - time)
        return norm.pdf(factor1 / factor2) / factor2

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        pass


class BinaryAssetPut(option.VanillaOption):
    """European asset-or-nothing put option in Bachelier model. Pays out
    one unit of the asset if the spot is below the strike at expiry."""

    def __init__(self, vol, strike, expiry):
        super().__init__(vol, strike, expiry)
        self._option_type = global_types.OptionType.BINARY_ASSET_PUT

    @property
    def option_type(self):
        return self._option_type

    def payoff(self, spot):
        return payoffs.binary_asset_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        pass

    def vega(self,
             spot: (float, np.ndarray),
             time: float) -> (float, np.ndarray):
        pass

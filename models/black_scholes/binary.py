import math
import numpy as np
from scipy.stats import norm

import models.options as options
import models.black_scholes.misc as misc
import models.black_scholes.sde as sde
import utils.global_types as global_types
import utils.payoffs as payoffs

# TODO: Add continuously compounded dividend yield


class BinaryCashCall(sde.SDE, options.VanillaOption):
    """European cash-or-nothing call option in Black-Scholes model.
    Pays out one unit of cash if the spot is above the strike at
    expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.BINARY_CASH_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """Price function."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(d2)

    def delta(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """1st order price sensitivity wrt the underlying state."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return math.exp(-self.rate * (self.expiry - time)) * norm.pdf(d2) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass


class BinaryAssetCall(sde.SDE, options.VanillaOption):
    """European asset-or-nothing call option in Black-Scholes model.
    Pays out one unit of the asset if the spot is above the strike at
    expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.BINARY_ASSET_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.binary_asset_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """Price function."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return spot * norm.cdf(d1)

    def delta(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """1st order price sensitivity wrt the underlying state."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return spot * norm.pdf(d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(d1)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass


class BinaryCashPut(sde.SDE, options.VanillaOption):
    """European cash-or-nothing put option in Black-Scholes model. Pays
    out one unit of cash if the spot is below the strike at expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.BINARY_CASH_PUT

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.binary_cash_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """Price function."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return math.exp(-self.rate * (self.expiry - time)) * norm.cdf(-d2)

    def delta(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """1st order price sensitivity wrt the underlying state."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return - math.exp(-self.rate * (self.expiry - time)) * norm.pdf(-d2) \
            / (spot * self.vol * math.sqrt(self.expiry - time))

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass


class BinaryAssetPut(sde.SDE, options.VanillaOption):
    """European asset-or-nothing put option in Black-Scholes model. Pays
    out one unit of the asset if the spot is below the strike at
    expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self.strike = strike
        self.expiry_idx = expiry_idx

        self.option_type = global_types.InstrumentType.BINARY_ASSET_PUT

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.binary_asset_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """Price function."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return spot * norm.cdf(-d1)

    def delta(self,
              spot: (float, np.ndarray),
              time_idx: int):
        """1st order price sensitivity wrt the underlying state."""
        time = self.event_grid[time_idx]
        d1, d2 = misc.d1d2(spot, time, self.rate, self.vol,
                           self.expiry, self.strike, self.dividend)
        return - spot * norm.pdf(-d1) \
            / (spot * self.vol * math.sqrt(self.expiry - time)) + norm.cdf(-d1)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        pass

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        pass

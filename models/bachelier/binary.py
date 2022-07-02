import math
import numpy as np
from scipy.stats import norm

import models.bachelier.option as option
import utils.global_types as global_types
import utils.payoffs as payoffs


class BinaryCashCall(option.VanillaOption):
    """European cash-or-nothing call option in Bachelier model. Pays out
    one unit of cash if the spot is above the strike at expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid, strike, expiry_idx)

        self._option_type = global_types.InstrumentType.BINARY_CASH_CALL

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_cash_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass


class BinaryAssetCall(option.VanillaOption):
    """European asset-or-nothing call option in Bachelier model. Pays
    out one unit of the asset if the spot is above the strike at expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid, strike, expiry_idx)

        self._option_type = global_types.InstrumentType.BINARY_ASSET_CALL

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_asset_call(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass


class BinaryCashPut(option.VanillaOption):
    """European cash-or-nothing put option in Bachelier model. Pays out
    one unit of cash if the spot is below the strike at expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid, strike, expiry_idx)

        self._option_type = global_types.InstrumentType.BINARY_CASH_PUT

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_cash_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass


class BinaryAssetPut(option.VanillaOption):
    """European asset-or-nothing put option in Bachelier model. Pays out
    one unit of the asset if the spot is below the strike at expiry.
    """

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid, strike, expiry_idx)

        self._option_type = global_types.InstrumentType.BINARY_ASSET_PUT

    @property
    def option_type(self) -> global_types.InstrumentType:
        return self._option_type

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """..."""
        return payoffs.binary_asset_put(spot, self.strike)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """..."""
        pass

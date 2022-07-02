import abc
import math
import numpy as np

import models.bachelier.sde as sde
import instruments.options as option


class VanillaOption(option.VanillaOption, sde.SDE):
    """Vanilla option in Bachelier model."""

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(rate, vol, event_grid)
        self._strike = strike
        self._expiry_idx = expiry_idx

    @property
    @abc.abstractmethod
    def option_type(self):
        pass

    @property
    def strike(self) -> float:
        return self._strike

    @strike.setter
    def strike(self, strike_):
        self._strike = strike_

    @property
    def expiry(self) -> float:
        return self.event_grid[self._expiry_idx]

    @property
    def expiry_idx(self) -> int:
        return self._expiry_idx

    @expiry_idx.setter
    def expiry_idx(self,
                   expiry_idx_: int):
        self._expiry_idx = expiry_idx_

    def dn(self,
           spot: (float, np.ndarray),
           time: float) -> (float, np.ndarray):
        """Factor in Bachelier formula."""
        # Time-to-maturity
        ttm = self.expiry - time
        return (spot - self.strike) / (self.vol * math.sqrt(ttm))

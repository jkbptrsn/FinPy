import abc
import numpy as np
from typing import Tuple

import models.black_scholes.sde as sde
import instruments.options as options


class VanillaOption(options.VanillaOption, sde.SDE):
    """Vanilla option in Black-Scholes model."""

    def __init__(self,
                 rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int,
                 dividend: float = 0):
        super().__init__(rate, vol, event_grid, dividend)
        self._strike = strike
        self._expiry_idx = expiry_idx

    @property
    @abc.abstractmethod
    def option_type(self) -> str:
        pass

    @property
    def strike(self) -> float:
        return self._strike

    @strike.setter
    def strike(self, strike_):
        self._strike = strike_

    @property
    def expiry(self) -> float:
        return self._event_grid[self._expiry_idx]

    @property
    def expiry_idx(self) -> int:
        return self._expiry_idx

    @expiry_idx.setter
    def expiry_idx(self, expiry_idx_):
        self._expiry_idx = expiry_idx_

    @abc.abstractmethod
    def payoff(self,
               state: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        pass

#    @abc.abstractmethod
#    def payoff_dds(self,
#                   state: (float, np.ndarray)) -> (float, np.ndarray):
#        """1st order partial derivative of payoff function wrt the
#        underlying state."""
#        pass

    @abc.abstractmethod
    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """Price function."""
        pass

    def d1d2(self,
             spot: (float, np.ndarray),
             time: float) \
            -> (Tuple[float, float], Tuple[np.ndarray, np.ndarray]):
        """Factors in Black-Scholes formula.
        - Returns Tuple[float, float] if spot is a float
        - Returns Tuple[np.ndarray, np.ndarray] otherwise
        """
        spot *= np.exp(-self.dividend * (self.expiry - time))
        d1 = np.log(spot / self._strike) \
            + (self.rate + self.vol ** 2 / 2) * (self.expiry - time)
        d1 /= self.vol * np.sqrt(self.expiry - time)
        return d1, d1 - self.vol * np.sqrt(self.expiry - time)

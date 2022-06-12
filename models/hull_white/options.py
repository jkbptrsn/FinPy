import abc
import numpy as np

import instruments.options as options
import models.hull_white.sde as sde
import utils.misc as misc


class VanillaOption(options.VanillaOption, sde.SDE):
    """Vanilla option in Hull-White model."""

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 forward_rate: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 strike: float,
                 expiry_idx: int):
        super().__init__(kappa, vol, forward_rate, event_grid)
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
    def strike(self,
               strike_: float):
        self._strike = strike_

    @property
    def expiry(self) -> int:
        return self._expiry_idx

    @expiry.setter
    def expiry(self,
               expiry_idx_: int):
        self._expiry_idx = expiry_idx_

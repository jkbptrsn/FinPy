import abc
import numpy as np

import instruments.bonds as bonds
import models.hull_white.sde as sde
import utils.misc as misc


class Bond(bonds.Bond, sde.SDE):
    """Bond class in Hull-White model."""

    def __init__(self,
                 kappa: misc.DiscreteFunc,
                 vol: misc.DiscreteFunc,
                 forward_rate: misc.DiscreteFunc,
                 event_grid: np.ndarray,
                 maturity_idx: int):
        super().__init__(kappa, vol, forward_rate, event_grid)
        self._maturity_idx = maturity_idx

        # Initialize SDE object
        self.initialization()

    @property
    @abc.abstractmethod
    def bond_type(self):
        pass

    @property
    def maturity(self) -> float:
        return self.event_grid[self._maturity_idx]

    @property
    def maturity_idx(self) -> int:
        return self._maturity_idx

    @maturity_idx.setter
    def maturity_idx(self,
                     maturity_idx_: int):
        self._maturity_idx = maturity_idx_

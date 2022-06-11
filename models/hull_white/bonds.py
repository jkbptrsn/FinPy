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
                 maturity: float):
        super().__init__(kappa, vol, forward_rate, event_grid)
        self._maturity = maturity

        # Initialize SDE object
        self.initialization()

    @property
    @abc.abstractmethod
    def bond_type(self):
        pass

    @property
    def maturity(self) -> float:
        return self._maturity

    @maturity.setter
    def maturity(self,
                 maturity_: float):
        self._maturity = maturity_

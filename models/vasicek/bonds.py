import abc

import instruments.bonds as bonds
import models.vasicek.sde as sde


class Bond(bonds.Bond, sde.SDE):
    """Bond class in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 maturity: float):
        super().__init__(kappa, mean_rate, vol)
        self._maturity = maturity

    @property
    @abc.abstractmethod
    def option_type(self):
        pass

    @property
    def maturity(self) -> float:
        return self._maturity

    @maturity.setter
    def maturity(self, maturity_):
        self._maturity = maturity_

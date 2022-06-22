import abc
import math
import numpy as np

import instruments.bonds as bonds
import models.vasicek.sde as sde


class Bond(bonds.Bond, sde.SDE):
    """Bond class in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 maturity: float):
        super().__init__(kappa, mean_rate, vol, event_grid)
        self._maturity = maturity

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


def a_factor(time1: float,
             time2: float,
             kappa: float,
             mean_rate: float,
             vol: float) -> float:
    """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
    vol_sq = vol ** 2
    four_kappa = 4 * kappa
    two_kappa_sq = 2 * kappa ** 2
    b = b_factor(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) \
        * (b - (time2 - time1)) - vol_sq * b ** 2 / four_kappa


def b_factor(time1: float,
             time2: float,
             kappa: float) -> float:
    """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
    return (1 - math.exp(- kappa * (time2 - time1))) / kappa


def dadt(time1: float,
         time2: float,
         kappa: float,
         mean_rate: float,
         vol: float) -> float:
    """Time derivative of A
    Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
    """
    vol_sq = vol ** 2
    two_kappa = 2 * kappa
    two_kappa_sq = 2 * kappa ** 2
    db = dbdt(time1, time2, kappa)
    return (mean_rate - vol_sq / two_kappa_sq) * (db + 1) \
        - vol_sq * b_factor(time1, time2, kappa) * db / two_kappa


def dbdt(time1: float,
         time2: float,
         kappa: float) -> float:
    """Time derivative of B
    Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
    """
    return - math.exp(- kappa * (time2 - time1))

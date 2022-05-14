import abc
import math

import instruments.bonds as bonds
import models.cox_ingersoll_ross.sde as sde


class Bond(bonds.Bond, sde.SDE):
    """Bond class in CIR model."""

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


def a_factor(time1: float,
             time2: float,
             kappa: float,
             mean_rate: float,
             vol: float) -> float:
    """Eq. (3.25), Brigo & Mercurio 2007."""
    h = math.sqrt(kappa ** 2 + 2 * vol ** 2)
    exp_kappa_h = math.exp((kappa + h) * (time2 - time1) / 2)
    exp_h = math.exp(h * (time2 - time1))
    exponent = 2 * kappa * mean_rate / vol ** 2
    return (2 * h * exp_kappa_h /
            (2 * h + (kappa + h) * (exp_h - 1))) ** exponent


def b_factor(time1: float,
             time2: float,
             kappa: float,
             vol: float) -> float:
    """Eq. (3.25), Brigo & Mercurio 2007."""
    h = math.sqrt(kappa ** 2 + 2 * vol ** 2)
    exp_h = math.exp(h * (time2 - time1))
    return 2 * (exp_h - 1) / (2 * h + (kappa + h) * (exp_h - 1))


def dadt(time1: float,
         time2: float,
         kappa: float,
         mean_rate: float,
         vol: float) -> float:
    """Time derivative of A: Eq. (3.25), Brigo & Mercurio 2007."""
    pass


def dbdt(time1: float,
         time2: float,
         kappa: float,
         vol: float) -> float:
    """Time derivative of B: Eq. (3.25), Brigo & Mercurio 2007."""
    pass

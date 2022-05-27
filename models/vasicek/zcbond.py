import numpy as np

import models.vasicek.bonds as bonds
import utils.global_types as global_types
import utils.payoffs as payoffs


class ZCBond(bonds.Bond):
    """Zero-coupon bond in Vasicek model."""

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 maturity: float):
        super().__init__(kappa, mean_rate, vol, maturity)
        self._bond_type = global_types.InstrumentType.ZERO_COUPON_BOND

    @property
    def bond_type(self) -> global_types.InstrumentType:
        return self._bond_type

    def a_factor(self,
                 time: float) -> float:
        """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
        return bonds.a_factor(time, self.maturity, self.kappa,
                              self.mean_rate, self.vol)

    def b_factor(self,
                 time: float) -> float:
        """Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010."""
        return bonds.b_factor(time, self.maturity, self.kappa)

    def dadt(self,
             time: float) -> float:
        """Time derivative of A
        Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return bonds.dadt(time, self.maturity, self.kappa,
                          self.mean_rate, self.vol)

    def dbdt(self,
             time: float) -> float:
        """Time derivative of B
        Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return bonds.dbdt(time, self.maturity, self.kappa)

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """Price function
        Proposition 10.1.4, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        return np.exp(self.a_factor(time) - self.b_factor(time) * spot)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        return - self.b_factor(time) * self.price(spot, time)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2st order price sensitivity wrt the underlying state."""
        return self.b_factor(time) ** 2 * self.price(spot, time)

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        return self.price(spot, time) \
            * (self.dadt(time) - self.dbdt(time) * spot)

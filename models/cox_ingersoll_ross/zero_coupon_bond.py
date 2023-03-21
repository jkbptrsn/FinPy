import numpy as np

from models import bonds
from models.cox_ingersoll_ross import misc
from models.cox_ingersoll_ross import sde
from utils import global_types
from utils import payoffs


class ZCBond(bonds.VanillaBondAnalytical1F):
    """Zero-coupon bond in CIR model.

    Attributes:
        kappa: Speed of mean-reversion.
        mean_rate: Mean interest rate.
        vol: Volatility.
        event_grid: Event dates, e.g. payment dates, represented as year
            fractions from the as-of date.
        maturity_idx: Maturity index on event_grid.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 event_grid: np.ndarray,
                 maturity_idx: int):
        super().__init__()
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.event_grid = event_grid
        self.maturity_idx = maturity_idx

        self.type = global_types.Instrument.ZERO_COUPON_BOND
        self.model = global_types.Model.VASICEK

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def a_factor(self,
                 time: float) -> float:
        """Eq. (3.25), Brigo & Mercurio 2007."""
        return misc.a_factor(time, self.maturity, self.kappa,
                             self.mean_rate, self.vol)

    def b_factor(self,
                 time: float) -> float:
        """Eq. (3.25), Brigo & Mercurio 2007."""
        return misc.b_factor(time, self.maturity, self.kappa, self.vol)

    def dadt(self,
             time: float) -> float:
        """Time derivative of A: Eq. (3.8), Brigo & Mercurio 2007."""
        return misc.dadt(time, self.maturity, self.kappa,
                         self.mean_rate, self.vol)

    def dbdt(self,
             time: float) -> float:
        """Time derivative of B: Eq. (3.8), Brigo & Mercurio 2007."""
        return misc.dbdt(time, self.maturity, self.kappa, self.vol)

    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function."""
        return payoffs.zero_coupon_bond(spot)

    def price(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """Price function: Eq. (3.25), Brigo & Mercurio 2007."""
        return np.exp(self.a_factor(time) - self.b_factor(time) * spot)

    def delta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state."""
        return - self.b_factor(time) * self.price(spot, time)

    def gamma(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state."""
        return self.b_factor(time) ** 2 * self.price(spot, time)

    def theta(self,
              spot: (float, np.ndarray),
              time: float) -> (float, np.ndarray):
        """1st order price sensitivity wrt time."""
        return self.price(spot, time) \
            * (self.dadt(time) - self.dbdt(time) * spot)

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.set_propagator()
            self.fd.propagation(dt)

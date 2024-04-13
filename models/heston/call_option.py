import cmath
import math
import typing

import numpy as np
import scipy

from models import options
from utils import global_types
from utils import payoffs


class Call(options.EuropeanOption2D):
    """European call option in Heston model.

    European call option written on stock price modelled by Heston SDE.

    Attributes:
        rate: Interest rate.
    """

    def __init__(self,
                 rate: float,
                 kappa: float,
                 eta: float,
                 vol: float,
                 correlation: float,
                 strike: float,
                 expiry_idx: int,
                 event_grid: np.ndarray):
        super().__init__()
        self.rate = rate
        self.kappa = kappa
        self.eta = eta
        self.vol = vol
        self.correlation = correlation
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid

        self.type = global_types.Instrument.EUROPEAN_CALL
        self.model = global_types.Model.HESTON

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        return payoffs.call(spot, self.strike)

    def price(self,
              spot_price: float,
              spot_variance: float,
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot_price: Current stock price.
            spot_variance: Current variance of stock price.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        # Time to maturity.
        tau = self.expiry - self.event_grid[event_idx]
        integral = self._integral(spot_price, spot_variance, tau)
        return spot_price \
            - self.strike * math.exp(-self.rate * tau) * integral / math.pi

    def _phi(self,
             u: float,
             spot_variance: float,
             tau: float):
        """Characteristic function for Heston model.

        See A. Lipton, Risk, 2002.

        Args:
            u: Independent variable.
            spot_variance: Current variance of stock price.
            tau: Time to maturity.

        Returns:
            Characteristic function.
        """
        alpha = -u * (u + 1j) / 2
        beta = self.kappa - 1j * u * self.vol * self.correlation
        gamma = self.vol ** 2 / 2
        d = cmath.sqrt(beta ** 2 - 4 * alpha * gamma)
        g = (beta - d) / (beta + d)
        h = cmath.exp(-d * tau)
        a = (beta - d) * tau - 2 * cmath.log((g * h - 1) / (g - 1))
        a = self.kappa * self.eta * a / self.vol ** 2
        b = (beta - d) * (1 - h) / (self.vol ** 2 * (1 - g * h))
        return cmath.exp(a + b * spot_variance)

    def _integral(self,
                  spot_price: float,
                  spot_variance: float,
                  tau: float):
        """Integral in call option pricing function.

        Args:
            spot_price: Current stock price.
            spot_variance: Current variance of stock price.
            tau: Time to maturity.

        Returns:
            Integral...
        """
        k = math.log(spot_price / self.strike) + self.rate * tau
        integrand = \
            (lambda u:
             np.real(np.exp((1j * u + 0.5) * k)
                     * self._phi(u - 0.5j, spot_variance, tau))
             / (u ** 2 + 0.25))
        integral, err = scipy.integrate.quad_vec(integrand, 0, np.inf)
        return integral

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
#            self.fd.set_propagator()
            self.fd.propagation(dt)

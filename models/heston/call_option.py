import cmath
import math
import typing

import numpy as np

from models import options
from utils import global_types
from utils import payoffs


class Call(options.Option2F):
    """European call option in Heston model.

    European call option written on stock price modelled by Heston SDE.

    Attributes:
        rate: Interest rate.
    """

    def __init__(
            self,
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

    def payoff(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current stock price.

        Returns:
            Payoff.
        """
        return payoffs.call(spot, self.strike)

    def price(
            self,
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

        forward_price = spot_price * math.exp(self.rate * tau)

        x = math.log(forward_price / self.strike)

        prop_0 = probability(0, x, spot_variance, self.kappa, self.eta, self.vol, self.correlation, tau)
        prop_1 = probability(1, x, spot_variance, self.kappa, self.eta, self.vol, self.correlation, tau)

        return self.strike * (math.exp(x) * prop_1 - prop_0) * math.exp(-self.rate * tau)

    def delta(self):
        pass

    def gamma(self):
        pass

    def theta(self):
        pass

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.propagation(dt)


def alpha(j, k):
    return -k ** 2 / 2 - 1j * k / 2 + 1j * j * k


def beta(j, k, lambda_, eta, rho):
    return lambda_ - rho * eta * j - 1j * rho * eta * k


def gamma(eta):
    return eta ** 2 / 2


def discriminant(j, k, lambda_, eta, rho):
    alpha_ = alpha(j, k)
    beta_ = beta(j, k, lambda_, eta, rho)
    gamma_ = gamma(eta)
    return cmath.sqrt(beta_ ** 2 - 4 * alpha_ * gamma_)


def r_func(sign, j, k, lambda_, eta, rho):
    beta_ = beta(j, k, lambda_, eta, rho)
    gamma_ = gamma(eta)
    d = discriminant(j, k, lambda_, eta, rho)
    if sign == "plus":
        return (beta_ + d) / (2 * gamma_)
    elif sign == "minus":
        return (beta_ - d) / (2 * gamma_)
    else:
        raise ValueError("Unknown sign.")


def g_func(j, k, lambda_, eta, rho):
    r_minus = r_func("minus", j, k, lambda_, eta, rho)
    r_plus = r_func("plus", j, k, lambda_, eta, rho)
    return r_minus / r_plus


def d_func(j, k, lambda_, eta, rho, tau):
    d = discriminant(j, k, lambda_, eta, rho)
    g = g_func(j, k, lambda_, eta, rho)
    r_minus = r_func("minus", j, k, lambda_, eta, rho)
    return r_minus * (1 - cmath.exp(-d * tau)) / (1 - g * cmath.exp(-d * tau))


def c_func(j, k, lambda_, eta, rho, tau):
    d = discriminant(j, k, lambda_, eta, rho)
    g = g_func(j, k, lambda_, eta, rho)
    r_minus = r_func("minus", j, k, lambda_, eta, rho)
    gamma_ = gamma(eta)
    result = 1 - g * cmath.exp(-d * tau)
    result /= 1 - g
    return lambda_ * (r_minus * tau - cmath.log(result) / gamma_)


def probability(j, x, variance, lambda_, theta, eta, rho, tau):
    n_steps = 101
    k_max = 100
    step_size = k_max / (n_steps - 1)
    integral = 0
    for i in range(n_steps):
        k = step_size * (i + 0.5)
        c = c_func(j, k, lambda_, eta, rho, tau)
        d = d_func(j, k, lambda_, eta, rho, tau)
        integrand = cmath.exp(c * theta + d * variance + 1j * k * x) / (1j * k)
        integral += integrand.real * step_size
    return 0.5 + integral / cmath.pi

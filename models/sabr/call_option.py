import math
import typing

import numpy as np
from scipy.stats import norm

from models import options
from models.black_scholes import misc
from utils import global_types
from utils import payoffs


class Call(options.Option2F):
    """European call option in SABR model.

    European call option written on stock price modelled by SABR SDE.

    Attributes:
        rate: Interest rate.
    """

    def __init__(
            self,
            rate: float,
            beta: float,
            vol: float,
            correlation: float,
            strike: float,
            expiry_idx: int,
            event_grid: np.ndarray):
        super().__init__()
        self.rate = rate
        self.beta = beta
        self.vol = vol
        self.correlation = correlation
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.event_grid = event_grid

        self.type = global_types.Instrument.EUROPEAN_CALL
        self.model = global_types.Model.SABR

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
        # Discount factor.
        discount = math.exp(-self.rate * tau)
        # Forward price.
        forward_price = spot_price / discount

        # Implied vol.
        f_mid = (spot_price + self.strike) / 2
#        f_mid = math.sqrt(spot_price * self.strike)
        c_f_mid = f_mid ** self.beta

        eta = spot_price ** (1 - self.beta) - self.strike ** (1 - self.beta)
        eta = self.vol * eta / (spot_variance * (1 - self.beta))

        gamma_1 = self.beta / f_mid
        gamma_2 = - self.beta * (1 - self.beta) / f_mid ** 2

        d_func = math.sqrt(1 - 2 * self.correlation * eta + eta ** 2)
        d_func += eta - self.correlation
        d_func = math.log(d_func / (1 - self.correlation))

        epsilon = self.vol ** 2 * tau

        term_1 = (2 * gamma_2 - gamma_1 ** 2 + 1 / f_mid ** 2) / 24
        term_1 *= (spot_variance * c_f_mid / self.vol) ** 2

        term_2 = self.correlation * gamma_1 * spot_variance * c_f_mid
        term_2 /= 4 * self.vol

        term_3 = (2 - 3 * self.correlation ** 2) / 24

        implied_vol = 1 + (term_1 + term_2 + term_3) * epsilon
        implied_vol *= self.vol * math.log(spot_price / self.strike) / d_func

        d1, d2 = misc.d1d2(spot_price, 0,
                           self.rate, implied_vol, self.expiry, self.strike)
        return spot_price * norm.cdf(d1) \
            - self.strike * norm.cdf(d2) * math.exp(-self.rate * tau)

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
#            self.fd.set_propagator()
            self.fd.propagation(dt)

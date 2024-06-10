import math
import typing

import numpy as np

from models import options
from models.heston import misc_european_option as misc
from utils import global_types
from utils import payoffs


class Call(options.Option2F):
    """European call option in Heston model.

    European call option written on stock price modelled by Heston SDE.

    Attributes:
        rate: Risk-free interest rate.
        kappa: Speed of mean reversion of variance.
        eta: Long-term mean of variance.
        vol: Volatility of variance.
        correlation: Correlation parameter.
        strike: Strike price of stock at expiry.
        expiry_idx: Expiry index on event grid.
        event_grid: Event dates as year fractions from as-of date.
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
        # Forward price.
        forward_price = spot_price * math.exp(self.rate * tau)
        # Log-moneyness.
        x = math.log(forward_price / self.strike)
        # Probabilities.
        prop_0 = misc.probability(0, x, spot_variance, self.kappa, self.eta,
                                  self.vol, self.correlation, tau)
        prop_1 = misc.probability(1, x, spot_variance, self.kappa, self.eta,
                                  self.vol, self.correlation, tau)
        return (self.strike * (math.exp(x) * prop_1 - prop_0)
                * math.exp(-self.rate * tau))

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        # Backward propagation.
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.propagation(dt)

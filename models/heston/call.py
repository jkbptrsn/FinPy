import typing

import numpy as np

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
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current stock price.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        pass

    def fd_solve(self):
        """Run solver on event_grid..."""
        for dt in np.flip(np.diff(self.event_grid)):
            self.fd.set_propagator()
            self.fd.propagation(dt)

import typing

import numpy as np

from models import options
from models.vasicek import misc
from models.vasicek import sde
from models.vasicek import zero_coupon_bond as zcbond
from utils import global_types
from utils import payoffs


class Call(options.EuropeanOptionAnalytical1F):
    """European call option in Vasicek model.

    European call option written on a zero-coupon bond.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        strike: Strike price of zero-coupon bond at expiry.
        expiry_idx: Expiry index on event grid.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates represented as year fractions from as-of
            date.
    """

    def __init__(self,
                 kappa: float,
                 mean_rate: float,
                 vol: float,
                 strike: float,
                 expiry_idx: int,
                 maturity_idx: int,
                 event_grid: np.ndarray):
        super().__init__()
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.strike = strike
        self.expiry_idx = expiry_idx
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid

        # Underlying zero-coupon bond.
        self.zcbond = zcbond.ZCBond(self.kappa, self.mean_rate, self.vol,
                                    self.maturity_idx, self.event_grid)

        self.model = global_types.Model.VASICEK
        self.type = global_types.Instrument.EUROPEAN_CALL

    @property
    def expiry(self) -> float:
        return self.event_grid[self.expiry_idx]

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current value of underlying zero-coupon bond.

        Returns:
            Payoff.
        """
        return payoffs.call(spot, self.strike)

    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return misc.european_option_price(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            "Call")

    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        return misc.european_option_delta(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            "Call")

    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        return misc.european_option_gamma(
            spot, event_idx, self.kappa, self.mean_rate, self.vol,
            self.strike, self.expiry_idx, self.maturity_idx, self.event_grid,
            "Call")

    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Current short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

    def fd_solve(self):
        """Run finite difference solver on event grid."""
        self.fd.set_propagator()
        # Set terminal condition.
        self.fd.solution = self.zcbond.payoff(self.fd.grid)
        for idx, dt in enumerate(np.flip(np.diff(self.event_grid))):
            # Expiry of call option.
            if idx == self.maturity_idx - self.expiry_idx:
                self.fd.solution = self.payoff(self.fd.solution)
            self.fd.propagation(dt)

    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        self.mc_exact = \
            sde.SDE(self.kappa, self.mean_rate, self.vol, self.event_grid)

    def mc_exact_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Args:
            spot: Short rate at as-of date.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of short rate and discount processes
            represented on event grid.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)

import math
import typing

import numpy as np

from models import bonds
from models.vasicek import misc
from models.vasicek import sde
from utils import global_types
from utils import payoffs


class ZCBond(bonds.Bond1FAnalytical):
    """Zero-coupon bond in Vasicek model.

    Zero-coupon bond price dependent on short rate modelled by Vasicek
    SDE. See Andersen & Piterbarg (2010), Proposition 10.1.4.

    Attributes:
        kappa: Speed of mean reversion.
        mean_rate: Mean reversion level.
        vol: Volatility.
        maturity_idx: Maturity index on event grid.
        event_grid: Event dates as year fractions from as-of date.
    """

    def __init__(
            self,
            kappa: float,
            mean_rate: float,
            vol: float,
            maturity_idx: int,
            event_grid: np.ndarray):
        super().__init__()
        self.kappa = kappa
        self.mean_rate = mean_rate
        self.vol = vol
        self.maturity_idx = maturity_idx
        self.event_grid = event_grid

        self.model = global_types.Model.VASICEK
        self.type = global_types.Instrument.ZERO_COUPON_BOND

    @property
    def maturity(self) -> float:
        return self.event_grid[self.maturity_idx]

    def payoff(
            self,
            spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot short rate.

        Returns:
            Payoff.
        """
        return payoffs.zero_coupon_bond(spot)

    def price(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        return np.exp(self.a_function(event_idx)
                      - self.b_function(event_idx) * spot)

    def delta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt short rate.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        return -self.b_function(event_idx) * self.price(spot, event_idx)

    def gamma(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt short rate.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        return self.b_function(event_idx) ** 2 * self.price(spot, event_idx)

    def theta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        return self.price(spot, event_idx) \
            * (self.dadt(event_idx) - self.dbdt(event_idx) * spot)

    def fd_solve(self) -> None:
        """Run finite difference solver on event grid."""
        # Backward propagation.
        time_steps = np.flip(np.diff(self.event_grid))
        for dt in time_steps:
            self.fd.propagation(dt)

    def mc_exact_setup(self) -> None:
        """Setup exact Monte-Carlo solver."""
        self.mc_exact = \
            sde.SdeExact(self.kappa, self.mean_rate, self.vol, self.event_grid)

    def mc_exact_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Exact discretization.

        Args:
            spot: Spot short rate.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        self.mc_exact.paths(spot, n_paths, rng, seed, antithetic)
        pv = self.mc_present_value(self.mc_exact)
        self.mc_exact.mc_estimate = pv.mean()
        self.mc_exact.mc_error = pv.std(ddof=1)
        self.mc_exact.mc_error /= math.sqrt(n_paths)

    def mc_euler_setup(self) -> None:
        """Setup Euler Monte-Carlo solver."""
        self.mc_euler = \
            sde.SdeEuler(self.kappa, self.mean_rate, self.vol, self.event_grid)

    def mc_euler_solve(
            self,
            spot: float,
            n_paths: int,
            rng: np.random.Generator = None,
            seed: int = None,
            antithetic: bool = False) -> None:
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Spot short rate.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Use antithetic sampling for variance reduction?
                Default is False.
        """
        self.mc_euler.paths(spot, n_paths, rng, seed, antithetic)
        pv = self.mc_present_value(self.mc_euler)
        self.mc_euler.mc_estimate = pv.mean()
        self.mc_euler.mc_error = pv.std(ddof=1)
        self.mc_euler.mc_error /= math.sqrt(n_paths)

    def a_function(self, event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.a_function(event_time, self.maturity, self.kappa,
                               self.mean_rate, self.vol)

    def b_function(self, event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.b_function(event_time, self.maturity, self.kappa)

    def dadt(self, event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.dadt(event_time, self.maturity, self.kappa, self.mean_rate,
                         self.vol)

    def dbdt(self, event_idx: int) -> float:
        event_time = self.event_grid[event_idx]
        return misc.dbdt(event_time, self.maturity, self.kappa)

    def mc_present_value(
            self,
            mc_object) -> np.ndarray:
        """Present value for each Monte-Carlo path."""
        return mc_object.discount_paths[self.maturity_idx]

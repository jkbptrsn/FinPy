import abc
import typing

import numpy as np

from numerics.fd.theta import theta as fd_theta


class BondAnalytical1F(metaclass=abc.ABCMeta):
    """Bond in 1-factor model with closed-form solution."""

    def __init__(self):
        # Solver objects.
        self.fd = None
        self.mc_exact = None
        self.mc_euler = None

    @property
    @abc.abstractmethod
    def maturity(self) -> float:
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def delta(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt value of underlying.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        pass

    @abc.abstractmethod
    def gamma(
            self,
            spot: typing.Union[float, np.ndarray],
            event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt value of underlying.

        Args:
            spot: Spot short rate.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        pass

    @abc.abstractmethod
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
        pass

    def fd_setup(
            self,
            x_grid: np.ndarray,
            band: str = "tri",
            equidistant: bool = False,
            theta_value: float = 0.5) -> None:
        """Setting up finite difference solver.

        Args:
            x_grid: Grid in spatial dimension.
            band: Tri- ("tri") or pentadiagonal ("penta") matrix
                representation of operators. Default is tridiagonal.
            equidistant: Is grid equidistant? Default is false.
            theta_value: Determines the specific method
                0   : Explicit method.
                0.5 : Crank-Nicolson method (default).
                1   : Fully implicit method.
        """
        fd_theta.setup_solver(self, x_grid, band, equidistant, theta_value)
        self.fd.initialization()

    def fd_update(
            self,
            event_idx: int = -1) -> None:
        """Update drift, diffusion and rate vectors.

        Args:
            event_idx: Index on event grid. Default is -1.
        """
        fd_theta.update(self, event_idx)

    @abc.abstractmethod
    def fd_solve(self) -> None:
        """Run finite difference solver on event grid."""
        pass

    @abc.abstractmethod
    def mc_exact_setup(self) -> None:
        """Setup exact Monte-Carlo solver."""
        pass

    @abc.abstractmethod
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

        Returns:
            Realizations of short rate and discount processes
            represented on event grid.
        """
        pass

    @abc.abstractmethod
    def mc_euler_setup(self) -> None:
        """Setup Euler Monte-Carlo solver."""
        pass

    @abc.abstractmethod
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

        Returns:
            Realizations of short rate and discount processes
            represented on event grid.
        """
        pass

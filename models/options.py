import abc
import typing

import numpy as np

from numerics.fd.theta import theta as fd_theta
from numerics.fd.adi import craig_sneyd as fd_craig


# TODO: Rename to Option1F
# TODO: Add Monte-Carlo related methods.
class AmericanOption(metaclass=abc.ABCMeta):
    """American option..."""

    def __init__(self):
        # Solver objects.
        self.fd = None
        self.mc = None
        self.mc_exact = None

    @property
    @abc.abstractmethod
    def expiry(self) -> float:
        pass

    @abc.abstractmethod
    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Value of underlying at as-of date.

        Returns:
            Payoff.
        """
        pass

    def fd_setup(self,
                 x_grid: np.ndarray,
                 form: str = "tri",
                 equidistant: bool = False,
                 theta_value: float = 0.5):
        """Setting up finite difference solver.

        Args:
            x_grid: Grid in spatial dimension.
            form: Tri- ("tri") or pentadiagonal ("penta") form. Default
                is tridiagonal.
            equidistant: Is grid equidistant? Default is false.
            theta_value: Determines the specific method:
                0   : Explicit method.
                0.5 : Crank-Nicolson method (default).
                1   : Fully implicit method.
        """
        self.fd = fd_theta.setup_solver(self, x_grid, form, equidistant,
                                        theta_value)
        self.fd.initialization()

    @abc.abstractmethod
    def fd_solve(self):
        """Run finite difference solver on event grid."""
        pass


class Option1FAnalytical(metaclass=abc.ABCMeta):
    """Option in 1-factor model with closed-form solution."""

    def __init__(self):
        # Solver objects.
        self.fd = None
        self.mc_exact = None
        self.mc_euler = None

    @abc.abstractmethod
    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Spot value of underlying.

        Returns:
            Payoff.
        """
        pass

    @abc.abstractmethod
    def price(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """Price function.

        Args:
            spot: Spot value of underlying.
            event_idx: Index on event grid.

        Returns:
            Price.
        """
        pass

    @abc.abstractmethod
    def delta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt value of underlying.

        Args:
            spot: Spot value of underlying.
            event_idx: Index on event grid.

        Returns:
            Delta.
        """
        pass

    @abc.abstractmethod
    def gamma(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """2nd order price sensitivity wrt value of underlying.

        Args:
            spot: Spot value of underlying.
            event_idx: Index on event grid.

        Returns:
            Gamma.
        """
        pass

    @abc.abstractmethod
    def theta(self,
              spot: typing.Union[float, np.ndarray],
              event_idx: int) -> typing.Union[float, np.ndarray]:
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot value of underlying.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

    def fd_setup(self,
                 x_grid: np.ndarray,
                 form: str = "tri",
                 equidistant: bool = False,
                 theta_value: float = 0.5):
        """Setting up finite difference solver.

        Args:
            x_grid: Grid in spatial dimension.
            form: Tri- ("tri") or pentadiagonal ("penta") form. Default
                is tridiagonal.
            equidistant: Is grid equidistant? Default is false.
            theta_value: Determines the specific method:
                0   : Explicit method.
                0.5 : Crank-Nicolson method (default).
                1   : Fully implicit method.
        """
        self.fd = fd_theta.setup_solver(self, x_grid, form, equidistant,
                                        theta_value)
        self.fd.initialization()

    def fd_update(self,
                  event_idx: int = -1):
        """Update drift, diffusion and rate vectors.

        Args:
            event_idx: Index on event grid. Default is -1.
        """
        fd_theta.update(self, event_idx)

    @abc.abstractmethod
    def fd_solve(self):
        """Run finite difference solver on event grid."""
        pass

#    @abc.abstractmethod
    def mc_exact_setup(self):
        """Setup exact Monte-Carlo solver."""
        pass

#    @abc.abstractmethod
    def mc_exact_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Exact discretization.

        Args:
            spot: Spot value of underlying.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of underlying process represented on event
            grid.
        """
        pass

#    @abc.abstractmethod
    def mc_euler_setup(self):
        """Setup Euler Monte-Carlo solver."""
        pass

#    @abc.abstractmethod
    def mc_euler_solve(self,
                       spot: float,
                       n_paths: int,
                       rng: np.random.Generator = None,
                       seed: int = None,
                       antithetic: bool = False):
        """Run Monte-Carlo solver on event grid.

        Euler-Maruyama discretization.

        Args:
            spot: Spot value of underlying.
            n_paths: Number of Monte-Carlo paths.
            rng: Random number generator. Default is None.
            seed: Seed of random number generator. Default is None.
            antithetic: Antithetic sampling for variance reduction.
                Default is False.

        Returns:
            Realizations of underlying process represented on event
            grid.
        """
        pass


# TODO: Rename to Option2F
class EuropeanOption2D(metaclass=abc.ABCMeta):
    """European option."""

    def __init__(self):
        # Solver objects.
        self.fd = None

    @property
    @abc.abstractmethod
    def expiry(self) -> float:
        pass

    @abc.abstractmethod
    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Value of underlying at as-of date.

        Returns:
            Payoff.
        """
        pass

    def fd_setup(self,
                 x_grid: np.ndarray,
                 y_grid: np.ndarray,
                 form: str = "tri",
                 equidistant: bool = False,
                 theta_value: float = 0.5):
        """Setting up finite difference solver.

        Args:
            x_grid: Grid in x dimension.
            y_grid: Grid in y dimension.
            form: Tri- ("tri") or pentadiagonal ("penta") form. Default
                is tridiagonal.
            equidistant: Is grid equidistant? Default is false.
            theta_value: Theta parameter.
        """
        self.fd = fd_craig.setup_solver(self, x_grid, y_grid, form,
                                        equidistant, theta_value)
        self.fd.initialization()

    @abc.abstractmethod
    def fd_solve(self):
        """Run solver on event_grid..."""
        pass

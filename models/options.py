import abc
import typing

import numpy as np

from numerics.fd.theta import theta as fd_theta
from numerics.fd.adi import craig_sneyd as fd_craig


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
            spot: Current value of underlying.

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


class EuropeanOptionAnalytical1F(metaclass=abc.ABCMeta):
    """European option with closed-form solution."""

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
            spot: Current value of underlying.

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
            spot: Current value of underlying.
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
            spot: Current value of underlying.
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
            spot: Current value of underlying.
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
            spot: Current value of underlying.
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

    @abc.abstractmethod
    def fd_solve(self):
        """Run finite difference solver on event_grid."""
        pass

    # @abc.abstractmethod
    # def mc_setup(self):
    #     """Setup Monte-Carlo solver."""
    #     pass
    #
    # @abc.abstractmethod
    # def mc_solve(self):
    #     """Run Monte-Carlo solver on event_grid."""
    #     pass


# Could rename this to VanillaOptionAnalytical
class VanillaOption(metaclass=abc.ABCMeta):
    """Abstract vanilla option class."""

    @abc.abstractmethod
    def payoff(self,
               spot: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Payoff function.

        Args:
            spot: Current value of underlying.

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
            spot: Current value of underlying.
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
            spot: Current value of underlying.
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
            spot: Current value of underlying.
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
            spot: Current value of underlying.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

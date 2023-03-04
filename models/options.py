import abc
import typing

import numpy as np

from numerics.fd.theta import theta as fd_theta


class EuropeanOption(metaclass=abc.ABCMeta):
    """European option."""

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


class EuropeanOptionAnalytical(metaclass=abc.ABCMeta):
    """European option with closed-form solution."""

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
            equidistant:
            theta_value: Theta parameter.
        """
        self.fd = fd_theta.setup_solver(self, x_grid, form, equidistant, theta_value)
        self.fd.initialization()

    @abc.abstractmethod
    def fd_solve(self):
        """Run solver on event_grid..."""
        pass


class VanillaOptionNew(metaclass=abc.ABCMeta):
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

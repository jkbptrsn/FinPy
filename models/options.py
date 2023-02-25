import abc
import numpy as np


class EuropeanOption(metaclass=abc.ABCMeta):
    """Abstract European option class."""

    @abc.abstractmethod
    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Current value of underlying.

        Returns:
            Payoff.
        """
        pass


class VanillaOptionNew(metaclass=abc.ABCMeta):
    """Abstract vanilla option class."""

    @abc.abstractmethod
    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
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
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Current value of underlying.

        Returns:
            Payoff.
        """
        pass

    @abc.abstractmethod
    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
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
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
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
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
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
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt time.

        Args:
            spot: Current value of underlying.
            event_idx: Index on event grid.

        Returns:
            Theta.
        """
        pass

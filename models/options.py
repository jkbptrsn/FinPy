import abc
import numpy as np


class VanillaOption(metaclass=abc.ABCMeta):
    """Abstract vanilla option class."""

    @abc.abstractmethod
    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Spot rate.

        Returns:
            Option payoff.
        """
        pass

    @abc.abstractmethod
    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Spot rate.
            event_idx: Index of event.

        Returns:
            Option price.
        """
        pass

    @abc.abstractmethod
    def delta(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt the underlying state.

        Args:
            spot: Spot rate.
            event_idx: Index of event.

        Returns:
            Option delta.
        """
        pass

    @abc.abstractmethod
    def gamma(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """2nd order price sensitivity wrt the underlying state.

        Args:
            spot: Spot rate.
            event_idx: Index of event.

        Returns:
            Option gamma.
        """
        pass

    @abc.abstractmethod
    def theta(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """1st order price sensitivity wrt time.

        Args:
            spot: Spot rate.
            event_idx: Index of event.

        Returns:
            Option theta.
        """
        pass

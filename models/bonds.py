import abc
import numpy as np


class Bond(metaclass=abc.ABCMeta):
    """Abstract bond class."""

    @abc.abstractmethod
    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Spot rate.

        Returns:
            Bond payoff.
        """
        pass

    @abc.abstractmethod
    def price(self,
              spot: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        """Price function.

        Args:
            spot: Spot rate.
            event_idx: Index of time on event grid...

        Returns:
            Bond price.
        """
        pass

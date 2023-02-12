import abc
import numpy as np


class VanillaBond(metaclass=abc.ABCMeta):
    """Abstract vanilla bond class."""

    @abc.abstractmethod
    def payoff(self,
               spot: (float, np.ndarray)) -> (float, np.ndarray):
        """Payoff function.

        Args:
            spot: Current rate.

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
            spot: Current rate.
            event_idx: Index of event.

        Returns:
            Bond price.
        """
        pass

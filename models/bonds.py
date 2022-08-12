import abc
import numpy as np


class Bond(metaclass=abc.ABCMeta):
    """Abstract bond class."""

    @abc.abstractmethod
    def payoff(self,
               spot_rate: (float, np.ndarray)) -> (float, np.ndarray):
        pass

    @abc.abstractmethod
    def price(self,
              spot_rate: (float, np.ndarray),
              event_idx: int) -> (float, np.ndarray):
        pass

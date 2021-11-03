import abc
import numpy as np


class AbstractSDE(metaclass=abc.ABCMeta):
    """Abstract Stochastic Differential Equation class."""

    @property
    @abc.abstractmethod
    def model_name(self):
        pass

    @abc.abstractmethod
    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate realization(s) of stochastic process."""
        pass

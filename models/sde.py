import abc
import numpy as np


class SDE(metaclass=abc.ABCMeta):
    """Abstract Stochastic Differential Equation class."""

    # ADD RNG IN PATHS!!!

    @abc.abstractmethod
    def paths(self,
              spot: float,
              n_paths: int,
              seed: int = None,
              antithetic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        pass

import abc

import numpy as np


class SDE(metaclass=abc.ABCMeta):
    """Abstract Stochastic Differential Equation class."""

    @abc.abstractmethod
    def paths(self,
              spot: float,
              n_paths: int,
              rng: np.random.Generator = None,
              seed: int = None,
              antithetic: bool = False) -> (np.ndarray, np.ndarray):
        pass

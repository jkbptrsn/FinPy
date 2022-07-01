import abc
import numpy as np
from typing import Tuple


class SDE(metaclass=abc.ABCMeta):
    """Abstract Stochastic Differential Equation class."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @abc.abstractmethod
    def paths(self,
              spot: float,
              n_paths: int,
              antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        pass

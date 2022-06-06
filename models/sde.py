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
    def path(self,
             spot: (float, np.ndarray),
             time: float,
             n_paths: int,
             antithetic: bool = False) -> (float, np.ndarray):
        """Generate path(s) at t = time."""
        pass

    @abc.abstractmethod
    def path_time_grid(self,
                       spot: (float, np.ndarray),
                       time_grid: np.ndarray) -> np.ndarray:
        """Generate one path represented on time_grid."""
        pass

#    @abc.abstractmethod
    def paths(self,
              spot: float,
              n_paths: int,
              antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths."""
        pass

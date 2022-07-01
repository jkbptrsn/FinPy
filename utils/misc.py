import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from typing import Tuple


class DiscreteFunc:
    """Interpolation and extrapolation of discrete function."""

    def __init__(self,
                 name: str,
                 time_grid: np.ndarray,
                 values: np.ndarray,
                 interp_scheme: str = "zero",
                 extrap_scheme: bool = True):
        self._name = name
        self._time_grid = time_grid
        self._values = values
        self._interp_scheme = interp_scheme
        self._extrap_scheme = extrap_scheme

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_grid(self) -> np.ndarray:
        return self._time_grid

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def interp_scheme(self) -> str:
        return self._interp_scheme

    @interp_scheme.setter
    def interp_scheme(self,
                      interp_scheme_: str):
        self._interp_scheme = interp_scheme_

    @property
    def extrap_scheme(self) -> bool:
        return self._extrap_scheme

    @extrap_scheme.setter
    def extrap_scheme(self,
                      extrap_scheme_: bool):
        self._extrap_scheme = extrap_scheme_

    def interpolation(self,
                      time_grid_new: (float, np.ndarray)) \
            -> (float, np.ndarray):
        """Interpolate (and extrapolate) on time_grid_new."""
        if self._extrap_scheme:
            extrap = "extrapolate"
        else:
            extrap = None
        f = interp1d(self._time_grid, self._values,
                     kind=self._interp_scheme, fill_value=extrap)
        return f(time_grid_new)


def trapz(grid: np.ndarray,
          function: np.ndarray) -> np.ndarray:
    """Trapezoidal integration for each step along the grid."""
    dx = np.diff(grid)
    return dx * (function[1:] + function[:-1]) / 2


def cholesky_2d(correlation: float,
                n_sets: int,
                seed: int = None,
                antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Sets of two correlated standard normal random variables using
    Cholesky decomposition. In the 2-D case, the transformation is
    simply: (x1, correlation * x1 + sqrt{1 - correlation ** 2} * x2)
    """
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    corr_matrix = np.linalg.cholesky(corr_matrix)
    x1 = normal_realizations(n_sets, seed, antithetic)
    x2 = normal_realizations(n_sets, seed, antithetic)
    return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
        corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2


def normal_realizations(n_realizations: int,
                        seed: int = None,
                        antithetic: bool = False) -> np.ndarray:
    """Realizations of a standard normal random variable."""
    if antithetic and n_realizations % 2 == 1:
        raise ValueError("In antithetic sampling, the number of "
                         "realizations should be even.")
    anti = 1
    if antithetic:
        anti = 2
    if type(seed) == int:
        np.random.seed(seed)
    realizations = norm.rvs(size=n_realizations // anti)
    if antithetic:
        realizations = np.append(realizations, -realizations)
    return realizations


def monte_carlo_error(values: np.ndarray) -> float:
    """TODO: check this formula -- divide by sqrt(n)?"""
    sample_mean = np.sum(values) / values.size
    sample_variance = np.sum((values - sample_mean) ** 2) / (values.size - 1)
    return math.sqrt(sample_variance) / math.sqrt(values.size)

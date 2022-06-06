import numpy as np
from scipy.interpolate import interp1d


def trapezoidal(grid: np.ndarray,
                function: np.ndarray) -> np.ndarray:
    """Trapezoidal integration for each step along the grid."""
    dx = np.diff(grid)
    return dx * (function[1:] + function[:-1]) / 2


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

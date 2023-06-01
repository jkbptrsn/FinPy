import typing

import numpy as np
from scipy.interpolate import interp1d


class DiscreteFunc:
    """Interpolation and extrapolation of a discrete function.

    Attributes:
        name: Name of the function.
        event_grid: Event grid in year fractions.
        values: Function value at each point on event grid.
        interp_scheme: Interpolation scheme.
            Default is flat interpolation ("zero"). Some other
            interpolation schemes are "linear", "quadratic", "cubic",
            etc. For more information, see the scipy documentation.
        extrap_scheme: Use corresponding extrapolation scheme.
            Default is True.
    """

    def __init__(self,
                 name: str,
                 event_grid: np.ndarray,
                 func_grid: np.ndarray,
                 interp_scheme: str = "zero",
                 extrap_scheme: bool = True):
        self.name = name
        self.event_grid = event_grid
        self.values = func_grid
        self.interp_scheme = interp_scheme
        self.extrap_scheme = extrap_scheme

    def interpolation(self,
                      interp_grid: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Interpolate (and extrapolate) on interp_event_grid.

        Args:
            interp_grid: Interpolation grid in year fractions.

        Returns:
            Function values on interpolation grid.
        """
        if self.extrap_scheme:
            extrap = "extrapolate"
        else:
            extrap = None
        f = interp1d(self.event_grid, self.values,
                     kind=self.interp_scheme, fill_value=extrap)
        return f(interp_grid)

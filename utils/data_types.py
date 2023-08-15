import typing

import numpy as np
from scipy.interpolate import interp1d


class DiscreteFunc:
    """Interpolation and extrapolation of a discrete function.

    Attributes:
        name: Name of the function.
        event_grid: Event dates as year fractions from as-of date.
        values: Function value at each point on event grid.
        interp_scheme: Interpolation scheme.
            * zero
            * linear
            * quadratic
            * cubic
            Default is flat interpolation ("zero").For more information,
            see the scipy documentation.
        extrap_scheme: Use corresponding extrapolation scheme.
            Default is True.
    """

    def __init__(self,
                 name: str,
                 event_grid: np.ndarray,
                 values: np.ndarray,
                 interp_scheme: str = "zero",
                 extrap_scheme: bool = True):
        self.name = name
        self.event_grid = event_grid
        self.values = values
        self.interp_scheme = interp_scheme
        self.extrap_scheme = extrap_scheme

        self.f_interpolation = None
        self._initialization()

    def _initialization(self):
        """Set up interpolation function."""
        if self.extrap_scheme:
            extrap = "extrapolate"
        else:
            extrap = None
        self.f_interpolation = \
            interp1d(self.event_grid, self.values,
                     kind=self.interp_scheme, fill_value=extrap)

    def interpolation(self,
                      interp_grid: typing.Union[float, np.ndarray]) \
            -> typing.Union[float, np.ndarray]:
        """Interpolate (and extrapolate) on interp_grid.

        Args:
            interp_grid: Interpolation grid in year fractions.

        Returns:
            Interpolated values.
        """
        return self.f_interpolation(interp_grid)

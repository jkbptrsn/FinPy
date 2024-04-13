import abc

import numpy as np


class ADI2D:
    """Alternating Direction Implicit (ADI) method in 2 dimensions.

    The general structure of the PDE is
        dV/dt + (L_x + L_y + L_xy)V = 0,
    where
        L_x = drift_x * d/dx + 1/2 * diffusion_x^2 * d^2/dx^2
            - 1/2 * rate,
        L_y = drift_y * d/dy + 1/2 * diffusion_y^2 * d^2/dy^2
            - 1/2 * rate,
        L_xy = coupling * diffusion_x * diffusion_y * d^2/(dx dy).

    See Andersen & Piterbarg (2010).

    Attributes:
        grid_x: 1D grid for x-dimension. Assumed ascending.
        grid_y: 1D grid for y-dimension. Assumed ascending.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
    """

    def __init__(self,
                 grid_x: np.ndarray,
                 grid_y: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.band = band
        self.equidistant = equidistant

        self.drift_x = None
        self.drift_y = None
        self.diff_x = None
        self.diff_y = None
        self.diff_x_sq = None
        self.diff_y_sq = None
        self.rate_x = None
        self.rate_y = None
        self.coupling = 0

    @property
    def grid_min(self) -> (float, float):
        return self.grid_x[0], self.grid_y[0]

    @property
    def grid_max(self) -> (float, float):
        return self.grid_x[-1], self.grid_y[-1]

    @property
    def nstates(self) -> (int, int):
        return self.grid_x.size, self.grid_y.size

    def set_drift(self,
                  drift_x: np.ndarray,
                  drift_y: np.ndarray) -> None:
        """Drift matrices defined by underlying process."""
        self.drift_x = drift_x
        self.drift_y = drift_y

    def set_diffusion(self,
                      diffusion_x: np.ndarray,
                      diffusion_y: np.ndarray) -> None:
        """Diffusion matrices defined by underlying process."""
        self.diff_x = diffusion_x
        self.diff_y = diffusion_y
        self.diff_x_sq = np.square(diffusion_x)
        self.diff_y_sq = np.square(diffusion_y)

    def set_rate(self,
                 rate_x: np.ndarray,
                 rate_y: np.ndarray) -> None:
        """Rate matrix defined by underlying process."""
        self.rate_x = rate_x
        self.rate_y = rate_y

    def set_coupling(self, coupling: float) -> None:
        """Coupling parameter."""
        self.coupling = coupling

    @abc.abstractmethod
    def propagation(self, dt: float) -> None:
        pass

    def delta(self) -> np.ndarray:
        """Finite difference calculation of delta."""
        pass

    def gamma(self) -> np.ndarray:
        """Finite difference calculation of gamma."""
        pass

    def theta(self, dt: float = None) -> np.ndarray:
        """Theta calculated by first order forward finite difference."""
        pass

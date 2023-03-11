import abc

import numpy as np


class Base2D:
    """2D alternating direction implicit scheme.

    The general structure of the PDE is
        dV/dt
            + drift_x * dV/dx
            + drift_y * dV/dy
            + 1/2 * diffusion_x^2 * d^2V/dx^2
            + 1/2 * diffusion_y^2 * d^2V/dy^2 = rate * V,
    where the underlying 2-dimensional Markov process reads
        dx(t) = drift_x(t, x(t), y(t)) * dt
            + diffusion_x(t, x(t), y(t)) * dW_x(t),
        dy(t) = drift_y(t, x(t), y(t)) * dt
            + diffusion_y(t, x(t), y(t)) * dW_y(t),
    and rate = rate(t, x(t), y(t)).

    Attributes:
        x_grid: Grid in 1st spatial dimension. Assumed ascending.
        y_grid: Grid in 2nd spatial dimension. Assumed ascending.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
    """

    def __init__(self,
                 x_grid: np.ndarray,
                 y_grid: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.band = band
        self.equidistant = equidistant

        self.solution = None
        self.drift = None
        self.diff_sq = None
        self.rate = None
        self.identity_x = None
        self.identity_y = None
        self.propagator_x = None
        self.propagator_y = None

    @property
    def xmin(self) -> (float, float):
        return self.x_grid[0], self.y_grid[0]

    @property
    def xmax(self) -> (float, float):
        return self.x_grid[-1], self.y_grid[-1]

    @property
    def nstates(self) -> (int, int):
        return self.x_grid.size, self.y_grid.size

    def set_drift(self, drift: np.ndarray):
        """Drift matrix defined by underlying process."""
        self.drift = drift

    def set_diffusion(self, diffusion: np.ndarray):
        """Squared diffusion matrix defined by underlying process."""
        self.diff_sq = np.square(diffusion)

    def set_rate(self, rate: np.ndarray):
        """Rate vector defined by underlying process."""
        self.rate = rate

    @abc.abstractmethod
    def set_propagator(self):
        pass

    @abc.abstractmethod
    def propagation(self, dt: float):
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

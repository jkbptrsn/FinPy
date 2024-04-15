import numpy as np
from scipy.linalg import solve_banded

from numerics.fd.adi import base_class
from numerics.fd import differential_operators as do
from numerics.fd import linear_algebra as la


class PeachmanRachford2D(base_class.ADI2D):
    """Peachman-Rachford ADI method in 2 dimensions.

    The general structure of the PDE is
        dV/dt + (L_x + L_y)V = 0,
    where
        L_x = drift_x * d/dx + 1/2 * diffusion_x^2 * d^2/dx^2
            - 1/2 * rate,
        L_y = drift_y * d/dy + 1/2 * diffusion_y^2 * d^2/dy^2
            - 1/2 * rate.

    See Andersen & Piterbarg (2010).

    Attributes:
        grid_x: 1D grid for x-dimension. Assumed ascending.
        grid_y: 1D grid for y-dimension. Assumed ascending.
        band: Tri- ("tri") or pentadiagonal ("penta") matrix
            representation of operators. Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
    """

    def __init__(self,
                 grid_x: np.ndarray,
                 grid_y: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False):
        super().__init__(grid_x, grid_y, band, equidistant)

        self.solution = None
        self.identity_x = None
        self.identity_y = None
        self.ddx = None
        self.ddy = None
        self.d2dx2 = None
        self.d2dy2 = None
        self.propagator_x = None
        self.propagator_y = None

    def initialization(self) -> None:
        """Initialization of identity and propagator matrices."""
        self.identity_x = la.identity_matrix(self.nstates[0], self.band)
        self.identity_y = la.identity_matrix(self.nstates[1], self.band)
        if self.equidistant:
            dx = self.grid_x[1] - self.grid_x[0]
            dy = self.grid_y[1] - self.grid_y[0]
            self.ddx = do.ddx_equidistant(self.nstates[0], dx, self.band)
            self.ddy = do.ddx_equidistant(self.nstates[1], dy, self.band)
            self.d2dx2 = do.d2dx2_equidistant(self.nstates[0], dx, self.band)
            self.d2dy2 = do.d2dx2_equidistant(self.nstates[1], dy, self.band)
        else:
            self.ddx = do.ddx(self.grid_x, self.band)
            self.ddy = do.ddx(self.grid_y, self.band)
            self.d2dx2 = do.d2dx2(self.grid_x, self.band)
            self.d2dy2 = do.d2dx2(self.grid_y, self.band)

    def set_propagator_x(self, idx: int) -> None:
        """Propagator in x-dimension as banded matrix."""
        term_1 = \
            la.dia_matrix_prod(self.drift_x[:, idx], self.ddx, self.band)
        term_2 = \
            la.dia_matrix_prod(self.diff_x_sq[:, idx], self.d2dx2, self.band)
        term_3 = \
            la.dia_matrix_prod(self.rate_x[:, idx], self.identity_x, self.band)
        self.propagator_x = term_1 + term_2 / 2 - term_3 / 2

    def set_propagator_y(self, idx: int) -> None:
        """Propagator in y-dimension as banded matrix."""
        term_1 = \
            la.dia_matrix_prod(self.drift_y[idx, :], self.ddy, self.band)
        term_2 = \
            la.dia_matrix_prod(self.diff_y_sq[idx, :], self.d2dy2, self.band)
        term_3 = \
            la.dia_matrix_prod(self.rate_y[idx, :], self.identity_y, self.band)
        self.propagator_y = term_1 + term_2 / 2 - term_3 / 2

    def propagation(self, dt: float) -> None:
        """Propagation of solution matrix for one time step dt."""
        if self.band == "tri":
            dimension = (1, 1)
        elif self.band == "penta":
            dimension = (2, 2)
        else:
            raise ValueError(
                f"{self.band}: Unknown banded matrix. Use tri or penta.")
        factor = dt / 2
        rhs = np.zeros(self.nstates)
        # First split; right-hand side.
        for idx in range(self.nstates[0]):
            self.set_propagator_y(idx)
            operator = self.identity_y + factor * self.propagator_y
            rhs[idx, :] = \
                la.matrix_col_prod(operator, self.solution[idx, :], self.band)
        # First split; left-hand side.
        for idx in range(self.nstates[1]):
            self.set_propagator_x(idx)
            operator = self.identity_x - factor * self.propagator_x
            self.solution[:, idx] = \
                solve_banded(dimension, operator, rhs[:, idx])
        # Second split; right-hand side.
        for idx in range(self.nstates[1]):
            self.set_propagator_x(idx)
            operator = self.identity_x + factor * self.propagator_x
            rhs[:, idx] = \
                la.matrix_col_prod(operator, self.solution[:, idx], self.band)
        # Second split; left-hand side.
        for idx in range(self.nstates[0]):
            self.set_propagator_y(idx)
            operator = self.identity_y - factor * self.propagator_y
            self.solution[idx, :] = \
                solve_banded(dimension, operator, rhs[idx, :])

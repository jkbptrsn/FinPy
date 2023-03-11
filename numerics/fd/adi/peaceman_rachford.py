import numpy as np
from scipy.linalg import solve_banded

from numerics.fd.adi import base_class
from numerics.fd.theta import differential_operators as do
from numerics.fd.theta import linear_algebra as la
from numerics.fd.theta import misc
from utils import global_types
from utils import payoffs


class PeachmanRachford2D(base_class.Base2D):
    """2D Peachman-Rachford alternating direction implicit scheme.

    ...

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
        super().__init__(x_grid, y_grid, band, equidistant)

        self.solution = None
        self.drift = None
        self.diff_sq = None
        self.rate = None
        self.identity_x = None
        self.identity_y = None
        self.ddx = None
        self.ddy = None
        self.d2dx2 = None
        self.d2dy2 = None
        self.propagator_x = None
        self.propagator_y = None

    def initialization(self):
        """Initialization of identity matrix and propagator matrix."""
        self.identity_x = la.identity_matrix(self.nstates[0], self.band)
        self.identity_y = la.identity_matrix(self.nstates[1], self.band)
        self.set_propagator()

    def set_propagator(self):
        """Propagator as banded matrix."""
        if self.equidistant:
            dx = self.x_grid[1] - self.x_grid[0]
            dy = self.y_grid[1] - self.y_grid[0]
            self.ddx = do.ddx_equidistant(self.nstates[0], dx, self.band)
            self.ddy = do.ddx_equidistant(self.nstates[1], dy, self.band)
            self.d2dx2 = do.d2dx2_equidistant(self.nstates[0], dx, self.band)
            self.d2dy2 = do.d2dx2_equidistant(self.nstates[1], dy, self.band)
        else:
            self.ddx = do.ddx(self.x_grid, self.band)
            self.ddy = do.ddx(self.y_grid, self.band)
            self.d2dx2 = do.d2dx2(self.x_grid, self.band)
            self.d2dy2 = do.d2dx2(self.y_grid, self.band)

    def set_propagator_x(self,
                         idx: int):
        """..."""
        self.propagator_x = \
            - la.dia_matrix_prod(self.rate[:, idx], self.identity_x, self.band) \
            + la.dia_matrix_prod(self.drift[:, idx], self.ddx, self.band) \
            + la.dia_matrix_prod(self.diff_sq[:, idx], self.d2dx2, self.band) / 2

    def set_propagator_y(self,
                         idx: int):
        """..."""
        self.propagator_y = \
            - la.dia_matrix_prod(self.rate[idx, :], self.identity_y, self.band) \
            + la.dia_matrix_prod(self.drift[idx, :], self.ddy, self.band) \
            + la.dia_matrix_prod(self.diff_sq[idx, :], self.d2dy2, self.band) / 2

    def propagation(self, dt: float):
        """Propagation of solution vector for one time step dt."""
        rhs = np.zeros(self.nstates)

        # First split.
        for idx in range(self.x_grid.size):
            self.set_propagator_y(idx)
            operator = self.identity_y + dt * self.propagator_y / 2
            rhs[idx, :] = la.matrix_col_prod(operator,
                                             self.solution[idx, :], self.band)
        for idx in range(self.y_grid.size):
            self.set_propagator_x(idx)
            operator = self.identity_x - dt * self.propagator_x / 2
            if self.band == "tri":
                self.solution[:, idx] = \
                    solve_banded((1, 1), operator, rhs[:, idx])
            elif self.band == "penta":
                self.solution[:, idx] = \
                    solve_banded((2, 2), operator, rhs[:, idx])
            else:
                raise ValueError("Form should be tri or penta...")

        # Second split.
        for idx in range(self.y_grid.size):
            self.set_propagator_x(idx)
            operator = self.identity_x + dt * self.propagator_x / 2
            rhs[:, idx] = la.matrix_col_prod(operator,
                                             self.solution[:, idx], self.band)
        for idx in range(self.x_grid.size):
            self.set_propagator_y(idx)
            operator = self.identity_y - dt * self.propagator_y / 2
            if self.band == "tri":
                self.solution[idx, :] = \
                    solve_banded((1, 1), operator, rhs[idx, :])
            elif self.band == "penta":
                self.solution[idx, :] = \
                    solve_banded((2, 2), operator, rhs[idx, :])
            else:
                raise ValueError("Form should be tri or penta...")

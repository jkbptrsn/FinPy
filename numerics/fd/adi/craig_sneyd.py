import numpy as np
from scipy.linalg import solve_banded

from numerics.fd.adi import base_class
from numerics.fd import differential_operators as do
from numerics.fd import linear_algebra as la
from utils import global_types
from utils import payoffs


class CraigSneyd2D(base_class.ADI2D):
    """Craig-Sneyd ADI method in 2 dimensions.

    The general structure of the PDE is
        dV/dt + (L_x + L_y + L_xy)V = 0,
    where
        L_x = drift_x * d/dx + 1/2 * diffusion_x^2 * d^2/dx^2
            - 1/2 * rate,
        L_y = drift_y * d/dy + 1/2 * diffusion_y^2 * d^2/dy^2
            - 1/2 * rate,
        L_xy = coupling * diffusion_x * diffusion_y * d^2/dxdy

    See L.B.G. Andersen & V.V. Piterbarg, Interest Rate Modeling, 2010.

    Attributes:
        grid_x: 1D grid for x-dimension. Assumed ascending.
        grid_y: 1D grid for y-dimension. Assumed ascending.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the form of the time derivative.
        lambda_parameter: TODO: Determines the inclusion of the mixed derivative...
    """

    def __init__(self,
                 grid_x: np.ndarray,
                 grid_y: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False,
                 theta_parameter: float = 0.5,
                 lambda_parameter: float = 0.5):
        super().__init__(grid_x, grid_y, band, equidistant)
        self.theta_parameter = theta_parameter
        self.lambda_parameter = lambda_parameter

        self.solution = None
        self.identity_x = None
        self.identity_y = None
        self.ddx = None
        self.ddy = None
        self.d2dx2 = None
        self.d2dy2 = None
        self.propagator_x = None
        self.propagator_y = None

        self.propagator_x_tmp = None
        self.propagator_y_tmp = None

    def initialization(self) -> None:
        """Initialization of identity and propagator matrices."""
        self.identity_x = la.identity_matrix(self.nstates[0], self.band)
        self.identity_y = la.identity_matrix(self.nstates[1], self.band)
        self.set_propagator()

    # TODO: Move diff operators to set_differential_operators
    def set_propagator(self) -> None:
        """Propagator as banded matrix."""
        if self.band == "tri":
            n_diagonals = 3
        elif self.band == "penta":
            n_diagonals = 5
        else:
            raise ValueError(
                f"{self.band}: "
                f"Unknown form of banded matrix. Use tri or penta.")
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

        self.propagator_x_tmp = \
            np.zeros((self.nstates[1], n_diagonals, self.nstates[0]))
        self.propagator_y_tmp = \
            np.zeros((self.nstates[0], n_diagonals, self.nstates[1]))

        for idx in range(self.nstates[1]):
            self.set_propagator_x(idx)
            self.propagator_x_tmp[idx] = self.propagator_x
        for idx in range(self.nstates[0]):
            self.set_propagator_y(idx)
            self.propagator_y_tmp[idx] = self.propagator_y

    # TODO: Save as numpy array of arrays...
    def set_propagator_x(self, idx: int) -> None:
        """Propagator in x-dimension as banded matrix."""
        term_1 = \
            la.dia_matrix_prod(self.drift_x[:, idx], self.ddx, self.band)
        term_2 = \
            la.dia_matrix_prod(self.diff_x_sq[:, idx], self.d2dx2, self.band)
        term_3 = \
            la.dia_matrix_prod(self.rate_x[:, idx], self.identity_x, self.band)
        self.propagator_x = term_1 + term_2 / 2 - term_3 / 2

    # TODO: Save as numpy array of arrays...
    def set_propagator_y(self, idx: int) -> None:
        """Propagator in y-dimension as banded matrix."""
        term_1 = \
            la.dia_matrix_prod(self.drift_y[idx, :], self.ddy, self.band)
        term_2 = \
            la.dia_matrix_prod(self.diff_y_sq[idx, :], self.d2dy2, self.band)
        term_3 = \
            la.dia_matrix_prod(self.rate_y[idx, :], self.identity_y, self.band)
        self.propagator_y = term_1 + term_2 / 2 - term_3 / 2

    # TODO: What about correlation (and diffusion product)?
    def d2dxdy(self, function: np.ndarray) -> np.ndarray:
        """2nd order mixed derivative of function."""
        if self.equidistant:
            dx = self.grid_x[1] - self.grid_x[0]
            dy = self.grid_y[1] - self.grid_y[0]
            do.d2dxdy_equidistant(function, dx, dy)
            return np.zeros(function.shape)
        else:
            do.d2dxdy(function, self.grid_x, self.grid_y)
            return np.zeros(function.shape)

    def propagation(self, dt: float) -> None:
        """Propagation of solution matrix for one time step dt."""
        if self.band == "tri":
            dimension = (1, 1)
        elif self.band == "penta":
            dimension = (2, 2)
        else:
            raise ValueError(
                f"{self.band}: "
                f"Unknown form of banded matrix. Use tri or penta.")
        # Predictor step, first split; right-hand side.
        x_term = np.zeros(self.nstates)
        for idx in range(self.nstates[1]):
            operator = self.identity_x \
                + (1 - self.theta_parameter) * dt * self.propagator_x_tmp[idx]
            x_term[:, idx] = \
                la.matrix_col_prod(operator, self.solution[:, idx], self.band)
        y_term = np.zeros(self.nstates)
        for idx in range(self.nstates[0]):
            operator = dt * self.propagator_y_tmp[idx]
            y_term[idx, :] = \
                la.matrix_col_prod(operator, self.solution[idx, :], self.band)
        xy_term = dt * self.d2dxdy(self.solution)
        tmp = x_term + y_term + xy_term
        # Predictor step, first split; left-hand side.
        for idx in range(self.nstates[1]):
            operator = self.identity_x \
                - self.theta_parameter * dt * self.propagator_x_tmp[idx]
            tmp[:, idx] = solve_banded(dimension, operator, tmp[:, idx])
        # Predictor step, second split; right-hand side.
        tmp -= self.theta_parameter * y_term
        # Predictor step, second split; left-hand side.
        for idx in range(self.nstates[0]):
            operator = self.identity_y \
                - self.theta_parameter * dt * self.propagator_y_tmp[idx]
            tmp[idx, :] = solve_banded(dimension, operator, tmp[idx, :])
        # Corrector step, first split; right-hand side.
        tmp = x_term + y_term + (1 - self.lambda_parameter) * xy_term \
            + self.lambda_parameter * dt * self.d2dxdy(tmp)
        # Corrector step, first split; left-hand side.
        for idx in range(self.nstates[1]):
            operator = self.identity_x \
                - self.theta_parameter * dt * self.propagator_x_tmp[idx]
            tmp[:, idx] = solve_banded(dimension, operator, tmp[:, idx])
        # Corrector step, second split; right-hand side.
        tmp -= self.theta_parameter * y_term
        # Corrector step, second split; left-hand side.
        for idx in range(self.nstates[0]):
            operator = self.identity_y \
                - self.theta_parameter * dt * self.propagator_y_tmp[idx]
            tmp[idx, :] = solve_banded(dimension, operator, tmp[idx, :])
        # Update solution.
        self.solution = tmp


def setup_solver(instrument,
                 x_grid: np.ndarray,
                 y_grid: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False,
                 theta_parameter: float = 0.5) -> CraigSneyd2D:
    """Setting up finite difference solver.

    Args:
        instrument: Instrument object.
        x_grid: Grid in x dimension.
        y_grid: Grid in y dimension.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the specific method

    Returns:
        Finite difference solver.
    """
    solver = CraigSneyd2D(x_grid, y_grid, band, equidistant, theta_parameter)
    # Model specifications.
    if instrument.model == global_types.Model.HESTON:
        drift_x = instrument.rate * solver.grid_x
        drift_y = instrument.kappa * (instrument.eta - solver.grid_y)
        diffusion_x = np.sqrt(solver.grid_y) * solver.grid_x
        diffusion_y = instrument.vol * np.sqrt(solver.grid_y)
        rate_x = instrument.rate + 0 * solver.grid_x
        rate_y = instrument.rate + 0 * solver.grid_y
    else:
        raise ValueError("Model is not recognized.")
    solver.set_drift(drift_x, drift_y)
    solver.set_diffusion(diffusion_x, diffusion_y)
    solver.set_rate(rate_x, rate_y)
    # Terminal solution to PDE. TODO: Use payoff method of instrument object...
    if instrument.type == global_types.Instrument.EUROPEAN_CALL:

        # For all y-values....
        solver.solution = payoffs.call(solver.grid_x, instrument.strike)

    else:
        raise ValueError("Instrument is not recognized.")
    return solver

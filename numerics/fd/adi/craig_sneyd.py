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

    See Andersen & Piterbarg (2010).

    Attributes:
        grid_x: 1D grid for x-dimension. Assumed ascending.
        grid_y: 1D grid for y-dimension. Assumed ascending.
        band: Tri- ("tri") or pentadiagonal ("penta") matrix
            representation of operators. Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the form of the time derivative.
        lambda_parameter: Determines the inclusion of the mixed
            derivative.
    """

    def __init__(
            self,
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

    def d2dxdy(self, function: np.ndarray) -> np.ndarray:
        """2nd order mixed derivative of function."""
        return do.d2dxdy(function, self.ddx, self.ddy, self.band, self.band)

    def propagation(self, dt: float) -> None:
        """Propagation of solution matrix for one time step dt."""
        if self.band == "tri":
            dimension = (1, 1)
        elif self.band == "penta":
            dimension = (2, 2)
        else:
            raise ValueError(
                f"{self.band}: Unknown banded matrix. Use tri or penta.")
        # Predictor step, first split; right-hand side.
        x_term = np.zeros(self.nstates)
        for idx in range(self.nstates[1]):
            self.set_propagator_x(idx)
            operator = self.identity_x \
                + (1 - self.theta_parameter) * dt * self.propagator_x
            x_term[:, idx] = \
                la.matrix_col_prod(operator, self.solution[:, idx], self.band)
        y_term = np.zeros(self.nstates)
        for idx in range(self.nstates[0]):
            self.set_propagator_y(idx)
            operator = dt * self.propagator_y
            y_term[idx, :] = \
                la.matrix_col_prod(operator, self.solution[idx, :], self.band)
        mixed_term = self.coupling \
            * self.diff_x * self.diff_y * self.d2dxdy(self.solution)
        xy_term = dt * mixed_term
        tmp = x_term + y_term + xy_term
        # Predictor step, first split; left-hand side.
        for idx in range(self.nstates[1]):
            self.set_propagator_x(idx)
            operator = self.identity_x \
                - self.theta_parameter * dt * self.propagator_x
            tmp[:, idx] = solve_banded(dimension, operator, tmp[:, idx])
        # Predictor step, second split; right-hand side.
        tmp -= self.theta_parameter * y_term
        # Predictor step, second split; left-hand side.
        for idx in range(self.nstates[0]):
            self.set_propagator_y(idx)
            operator = self.identity_y \
                - self.theta_parameter * dt * self.propagator_y
            tmp[idx, :] = solve_banded(dimension, operator, tmp[idx, :])
        # Corrector step, first split; right-hand side.
        mixed_term = self.coupling \
            * self.diff_x * self.diff_y * self.d2dxdy(tmp)
        tmp = x_term + y_term + (1 - self.lambda_parameter) * xy_term \
            + self.lambda_parameter * dt * mixed_term
        # Corrector step, first split; left-hand side.
        for idx in range(self.nstates[1]):
            self.set_propagator_x(idx)
            operator = self.identity_x \
                - self.theta_parameter * dt * self.propagator_x
            tmp[:, idx] = solve_banded(dimension, operator, tmp[:, idx])
        # Corrector step, second split; right-hand side.
        tmp -= self.theta_parameter * y_term
        # Corrector step, second split; left-hand side.
        for idx in range(self.nstates[0]):
            self.set_propagator_y(idx)
            operator = self.identity_y \
                - self.theta_parameter * dt * self.propagator_y
            tmp[idx, :] = solve_banded(dimension, operator, tmp[idx, :])
        # Update solution.
        self.solution = tmp


def setup_solver(
        instrument,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        band: str = "tri",
        equidistant: bool = False,
        theta_parameter: float = 0.5) -> None:
    """Setting up finite difference solver.

    Args:
        instrument: Instrument object.
        x_grid: Grid in x dimension.
        y_grid: Grid in y dimension.
        band: Tri- ("tri") or pentadiagonal ("penta") matrix
            representation of operators. Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the specific method

    Returns:
        Finite difference solver.
    """
    instrument.fd = (
        CraigSneyd2D(x_grid, y_grid, band, equidistant, theta_parameter))
    update(instrument)
    # Terminal solution to PDE. TODO: Use payoff method of instrument object?
    if instrument.type == global_types.Instrument.EUROPEAN_CALL:
        solution = payoffs.call(instrument.fd.grid_x, instrument.strike)
        instrument.fd.solution = (
            np.outer(solution, np.ones(instrument.fd.grid_y.size)))
    else:
        raise ValueError("Unknown instrument.")


def update(
        instrument,
        event_idx: int = -1) -> None:
    """Update drift, diffusion and rate vectors.

    Args:
        instrument: Instrument object.
        event_idx: Index on event grid. Default is -1.
    """
    if instrument.model == global_types.Model.HESTON:
        drift_x = instrument.rate * instrument.fd.grid_x
        drift_x = np.outer(drift_x, np.ones(instrument.fd.grid_y.size))
        drift_y = instrument.kappa * (instrument.eta - instrument.fd.grid_y)
        drift_y = np.outer(np.ones(instrument.fd.grid_x.size), drift_y)
        diffusion_x = (
            np.outer(instrument.fd.grid_x, np.sqrt(instrument.fd.grid_y)))
        diffusion_y = (
            instrument.vol * np.sqrt(instrument.fd.grid_y))
        diffusion_y = (
            np.outer(np.ones(instrument.fd.grid_x.size), diffusion_y))
        rate_x = instrument.rate + 0 * instrument.fd.grid_x
        rate_x = np.outer(rate_x, np.ones(instrument.fd.grid_y.size))
        rate_y = instrument.rate + 0 * instrument.fd.grid_y
        rate_y = np.outer(np.ones(instrument.fd.grid_x.size), rate_y)
        instrument.fd.coupling = instrument.correlation

    # TODO: NM delete
    elif instrument.model == global_types.Model.SABR:
        drift_x = instrument.rate * instrument.fd.grid_x
        drift_x = np.outer(drift_x, np.ones(instrument.fd.grid_y.size))
        drift_y = 0 * instrument.fd.grid_y
        drift_y = np.outer(np.ones(instrument.fd.grid_x.size), drift_y)

        # Remember discount factor, D^(1-beta)...
        diffusion_x = np.power(instrument.fd.grid_x, instrument.beta)
        diffusion_x = np.outer(diffusion_x, instrument.fd.grid_y)
        diffusion_y = instrument.vol * instrument.fd.grid_y
        diffusion_y = np.outer(np.ones(instrument.fd.grid_x.size), diffusion_y)

        rate_x = instrument.rate + 0 * instrument.fd.grid_x
        rate_x = np.outer(rate_x, np.ones(instrument.fd.grid_y.size))
        rate_y = instrument.rate + 0 * instrument.fd.grid_y
        rate_y = np.outer(np.ones(instrument.fd.grid_x.size), rate_y)
        instrument.fd.coupling = instrument.correlation

    else:
        raise ValueError(f"Unknown model: {instrument.model}")
    instrument.fd.set_drift(drift_x, drift_y)
    instrument.fd.set_diffusion(diffusion_x, diffusion_y)
    instrument.fd.set_rate(rate_x, rate_y)

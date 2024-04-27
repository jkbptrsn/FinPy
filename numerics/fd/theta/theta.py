import abc

import numpy as np
from scipy.linalg import solve_banded

from numerics.fd import delta_gamma as greeks
from numerics.fd import differential_operators as do
from numerics.fd import linear_algebra as la
from utils import global_types


class ThetaBase:
    """Theta method for solving parabolic 1-factor PDE (base class).

    The general structure of the PDE is
        dV/dt + drift * dV/dx + 1/2 * diffusion^2 * d^2V/dx^2 = rate * V,
    where the underlying 1-dimensional Markov process reads
        dx_t = drift(t, x_t) * dt + diffusion(t, x_t) * dW_t.

    Attributes:
        grid: Grid in spatial dimension. Assumed ascending.
        band: Tri- ("tri") or pentadiagonal ("penta") matrix
            representation of operators. Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
    """

    def __init__(self,
                 grid: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False):
        self.grid = grid
        self.band = band
        self.equidistant = equidistant

        self.vec_solution = None
        self.vec_drift = None
        self.vec_diff_sq = None
        self.vec_rate = None
        self.mat_identity = None
        self.mat_propagator = None

    @property
    def xmin(self) -> float:
        return self.grid[0]

    @property
    def xmax(self) -> float:
        return self.grid[-1]

    @property
    def nstates(self) -> int:
        return self.grid.size

    @property
    def solution(self) -> np.ndarray:
        return self.vec_solution

    @solution.setter
    def solution(self, val: np.ndarray):
        self.vec_solution = val

    def set_drift(self, drift: np.ndarray):
        """Drift vector defined by underlying process."""
        self.vec_drift = drift

    def set_diffusion(self, diffusion: np.ndarray):
        """Squared diffusion vector defined by underlying process."""
        self.vec_diff_sq = np.square(diffusion)

    def set_rate(self, rate: np.ndarray):
        """Rate vector defined by underlying process."""
        self.vec_rate = rate

    @abc.abstractmethod
    def set_propagator(self):
        """Propagator as banded matrix."""
        pass

    @abc.abstractmethod
    def propagation(self, dt: float):
        """Propagation of solution vector for one time step dt."""
        pass

    def delta(self) -> np.ndarray:
        """Finite difference calculation of delta."""
        if self.equidistant:
            dx = self.grid[1] - self.grid[0]
            return greeks.delta_equidistant(dx, self.vec_solution, self.band)
        else:
            return greeks.delta(self.grid, self.vec_solution, self.band)

    def gamma(self) -> np.ndarray:
        """Finite difference calculation of gamma."""
        if self.equidistant:
            dx = self.grid[1] - self.grid[0]
            return greeks.gamma_equidistant(dx, self.vec_solution, self.band)
        else:
            return greeks.gamma(self.grid, self.vec_solution, self.band)

    @abc.abstractmethod
    def theta(self, dt: float = None) -> np.ndarray:
        """Finite difference calculation of theta."""
        pass

    def theta_calc(self, dt: float) -> np.ndarray:
        """Theta calculated by first order forward finite difference."""
        self.set_propagator()
        # Save current solution.
        solution_copy = self.solution.copy()
        # Forward propagation.
        self.propagation(-dt)
        forward = self.solution.copy()
        # Restore current solution.
        self.solution = solution_copy
        return (forward - self.solution) / dt


class Theta(ThetaBase):
    """Theta method for solving parabolic 1-factor PDE.

    The general structure of the PDE is
        dV/dt + drift * dV/dx + 1/2 * diffusion^2 * d^2V/dx^2 = rate * V,
    where the underlying 1-dimensional Markov process reads
        dx_t = drift(t, x_t) * dt + diffusion(t, x_t) * dW_t.

    The propagator is defined as
        dV/dt = - Propagator * V.

    See Andersen & Piterbarg (2010).

    scipy.linalg.solve_banded solves equation A x = b using standard LU
    factorization of A.

    Attributes:
        grid: Grid in spatial dimension.
        band: Tri- ("tri") or pentadiagonal ("penta") matrix
            representation of operators. Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the specific method.
            0   : Explicit method
            0.5 : Crank-Nicolson method (default)
            1   : Fully implicit method

    TODO: Smoothing of payoff functions? Not necessary according to Kwant Daddy
    TODO: Rannacher (initial) stepping? Not necessary according to Kwant Daddy
    """

    def __init__(
            self,
            grid: np.ndarray,
            band: str = "tri",
            equidistant: bool = False,
            theta_parameter: float = 0.5):
        super().__init__(grid, band, equidistant)
        self.theta_parameter = theta_parameter

        self.ddx = None
        self.d2dx2 = None

        self.dt_last = None

    def initialization(self) -> None:
        """Initialization of identity and propagator matrices."""
        self.mat_identity = la.identity_matrix(self.nstates, self.band)
        if self.equidistant:
            dx = self.grid[1] - self.grid[0]
            self.ddx = do.ddx_equidistant(self.nstates, dx, self.band)
            self.d2dx2 = do.d2dx2_equidistant(self.nstates, dx, self.band)
        else:
            self.ddx = do.ddx(self.grid, self.band)
            self.d2dx2 = do.d2dx2(self.grid, self.band)
        self.set_propagator()

    def set_propagator(self) -> None:
        """Propagator as banded matrix."""
        self.mat_propagator = \
            - la.dia_matrix_prod(self.vec_rate, self.mat_identity, self.band) \
            + la.dia_matrix_prod(self.vec_drift, self.ddx, self.band) \
            + la.dia_matrix_prod(self.vec_diff_sq, self.d2dx2, self.band) / 2

    def propagation(
            self,
            dt: float,
            time_dependent: bool = False) -> None:
        """Propagation of solution vector for one time step dt."""
        rhs = self.mat_identity \
            + (1 - self.theta_parameter) * dt * self.mat_propagator
        rhs = la.matrix_col_prod(rhs, self.vec_solution, self.band)
        if time_dependent:
            self.set_propagator()
        lhs = self.mat_identity \
            - self.theta_parameter * dt * self.mat_propagator
        if self.band == "tri":
            self.vec_solution = solve_banded((1, 1), lhs, rhs)
        elif self.band == "penta":
            self.vec_solution = solve_banded((2, 2), lhs, rhs)
        else:
            raise ValueError(
                f"{self.band}: Unknown banded matrix. Use tri or penta.")
        # Last used time step.
        self.dt_last = dt

    def theta(self, dt: float = None) -> np.ndarray:
        """Finite difference calculation of theta."""
        if not dt:
            if not self.dt_last:
                raise ValueError("Specify dt.")
            dt = self.dt_last
        return self.theta_calc(dt)


def setup_solver(
        instrument,
        grid: np.ndarray,
        band: str = "tri",
        equidistant: bool = False,
        theta_parameter: float = 0.5) -> None:
    """Setting up finite difference solver.

    Args:
        instrument: Instrument object.
        grid: Grid in spatial dimension.
        band: Tri- ("tri") or pentadiagonal ("penta") matrix
            representation of operators. Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the specific method:
            0   : Explicit method.
            0.5 : Crank-Nicolson method (default).
            1   : Fully implicit method.
    """
    instrument.fd = Theta(grid, band, equidistant, theta_parameter)
    update(instrument)
    # Terminal solution of PDE. TODO: Move to instrument class?
    instrument.fd.solution = instrument.payoff(instrument.fd.grid)


def update(
        instrument,
        event_idx: int = -1) -> None:
    """Update drift, diffusion and rate vectors.

    Args:
        instrument: Instrument object.
        event_idx: Index on event grid. Default is -1.
    """
    if instrument.model == global_types.Model.BLACK_SCHOLES:
        drift = instrument.rate * instrument.fd.grid
        diffusion = instrument.vol * instrument.fd.grid
        rate = instrument.rate + 0 * instrument.fd.grid

    # TODO: NM delete
    elif instrument.model == global_types.Model.BACHELIER:
        drift = 0 * instrument.fd.grid
        diffusion = instrument.vol + 0 * instrument.fd.grid
        rate = instrument.rate + 0 * instrument.fd.grid
    elif instrument.model == global_types.Model.CIR:
        drift = instrument.kappa * (instrument.mean_rate - instrument.fd.grid)
        diffusion = instrument.vol * np.sqrt(instrument.fd.grid)
        rate = instrument.fd.grid

    elif instrument.model == global_types.Model.HULL_WHITE_1F:
        if instrument.transformation == global_types.Transformation.ANDERSEN:
            drift = instrument.y_eg[event_idx] \
                - instrument.kappa_eg[event_idx] * instrument.fd.grid
            diffusion = instrument.vol_eg[event_idx] + 0 * instrument.fd.grid
            rate = instrument.fd.grid
        elif instrument.transformation == global_types.Transformation.PELSSER:
            drift = -instrument.kappa_eg[event_idx] * instrument.fd.grid
            diffusion = instrument.vol_eg[event_idx] + 0 * instrument.fd.grid
            rate = instrument.fd.grid
        else:
            raise ValueError(
                f"Unknown transformation: {instrument.transformation}")
    elif instrument.model == global_types.Model.VASICEK:
        drift = instrument.kappa * (instrument.mean_rate - instrument.fd.grid)
        diffusion = instrument.vol + 0 * instrument.fd.grid
        rate = instrument.fd.grid
    else:
        raise ValueError(f"Unknown model: {instrument.model}")
    instrument.fd.set_drift(drift)
    instrument.fd.set_diffusion(diffusion)
    instrument.fd.set_rate(rate)

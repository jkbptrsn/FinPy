import abc

import math
import numpy as np
from scipy.linalg import solve_banded

from numerics.fd import differential_operators as do
from numerics.fd import linear_algebra as la
from numerics.fd import delta_gamma as greeks
from utils import global_types


class ThetaBase:
    """Theta method for solving parabolic 1-factor PDE (base class).

    The general structure of the PDE is
        dV/dt + drift * dV/dx + 1/2 * diffusion^2 * d^2V/dx^2 = rate * V,
    TODO: What about the RHS for Ornstein-Uhlenbeck processes?
    where the underlying 1-dimensional Markov process reads
        dx_t = drift(t, x_t) * dt + diffusion(t, x_t) * dW_t.

    Attributes:
        grid: Grid in spatial dimension. Assumed ascending.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.

    TODO: Smoothing of payoff functions -- not necessary according to Andreasen
    TODO: Rannacher time stepping with fully implicit method -- not necessary according to Andreasen
    TODO: Upwinding -- rarely used by Andreasen
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
        pass

    @abc.abstractmethod
    def propagation(self, dt: float):
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
        pass

    def theta_calc(self, dt: float) -> np.ndarray:
        """Theta calculated by first order forward finite difference."""
        self.set_propagator()
        # Save current solution
        solution_copy = self.solution.copy()
        # Forward propagation
        self.propagation(-dt)
        forward = self.solution.copy()
        # Restore current solution
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

    scipy.linalg.solve_banded solves equation A x = b using standard LU
    factorization of A.

    Attributes:
        grid: Grid in spatial dimension.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the specific method.
            0   : Explicit method
            0.5 : Crank-Nicolson method (default)
            1   : Fully implicit method
    """

    def __init__(self,
                 grid: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False,
                 theta_parameter: float = 0.5):
        super().__init__(grid, band, equidistant)
        self.theta_parameter = theta_parameter

        self.dt_last = None

    def initialization(self):
        """Initialization of identity and propagator matrices."""
        self.mat_identity = la.identity_matrix(self.nstates, self.band)
        self.set_propagator()

    def set_propagator(self):
        """Propagator as banded matrix."""
        if self.equidistant:
            dx = self.grid[1] - self.grid[0]
            ddx = do.ddx_equidistant(self.nstates, dx, self.band)
            d2dx2 = do.d2dx2_equidistant(self.nstates, dx, self.band)
        else:
            ddx = do.ddx(self.grid, self.band)
            d2dx2 = do.d2dx2(self.grid, self.band)
        self.mat_propagator = \
            - la.dia_matrix_prod(self.vec_rate, self.mat_identity, self.band) \
            + la.dia_matrix_prod(self.vec_drift, ddx, self.band) \
            + la.dia_matrix_prod(self.vec_diff_sq, d2dx2, self.band) / 2

    def propagation(self,
                    dt: float,
                    time_dependent: bool = False):
        """Propagation of solution vector for one time step dt."""
        rhs = self.mat_identity \
            + (1 - self.theta_parameter) * dt * self.mat_propagator
        rhs = la.matrix_col_prod(rhs, self.vec_solution, self.band)

        if time_dependent:
            self.set_propagator()
            # TODO: Update propagator, if drift/diffusion is time-dependent.
            #  But then one would also need to update vec_drift and vec_diff_sq...

        lhs = self.mat_identity \
            - self.theta_parameter * dt * self.mat_propagator
        if self.band == "tri":
            self.vec_solution = solve_banded((1, 1), lhs, rhs)
        elif self.band == "penta":
            self.vec_solution = solve_banded((2, 2), lhs, rhs)
        else:
            raise ValueError("Form should be tri or penta...")
        # Last used time step.
        self.dt_last = dt

    def theta(self, dt: float = None) -> np.ndarray:
        """Finite difference calculation of theta."""
        if not dt:
            if not self.dt_last:
                raise ValueError("Specify dt.")
            dt = self.dt_last
        return self.theta_calc(dt)


def setup_solver(instrument,
                 grid: np.ndarray,
                 band: str = "tri",
                 equidistant: bool = False,
                 theta_parameter: float = 0.5) -> Theta:
    """Setting up finite difference solver.

    Args:
        instrument: Instrument object.
        grid: Grid in spatial dimension.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.
        equidistant: Is grid equidistant? Default is false.
        theta_parameter: Determines the specific method:
            0   : Explicit method.
            0.5 : Crank-Nicolson method (default).
            1   : Fully implicit method.

    Returns:
        Finite difference solver.
    """
    solver = Theta(grid, band, equidistant, theta_parameter)
    if instrument.model == global_types.Model.BACHELIER:
        drift = 0 * solver.grid
        diffusion = instrument.vol + 0 * solver.grid
        rate = instrument.rate + 0 * solver.grid
    elif instrument.model == global_types.Model.BLACK_SCHOLES:
        drift = instrument.rate * solver.grid
        diffusion = instrument.vol * solver.grid
        rate = instrument.rate + 0 * solver.grid
    elif instrument.model == global_types.Model.CIR:
        drift = instrument.kappa * (instrument.mean_rate - solver.grid)
        diffusion = instrument.vol * np.sqrt(solver.grid)
        rate = solver.grid
    elif instrument.model == global_types.Model.HULL_WHITE_1F:
        drift = instrument.y_eg[-1] - instrument.kappa_eg[-1] * solver.grid
        diffusion = instrument.vol_eg[-1] + 0 * solver.grid
        rate = solver.grid + instrument.forward_rate_eg[-1]
    elif instrument.model == global_types.Model.VASICEK:
        drift = instrument.kappa * (instrument.mean_rate - solver.grid)
        diffusion = instrument.vol + 0 * solver.grid
        rate = solver.grid
    else:
        raise ValueError(f"Model is not recognized: {instrument.model}")
    solver.set_drift(drift)
    solver.set_diffusion(diffusion)
    solver.set_rate(rate)

    # Terminal solution to PDE. TODO: Move to each instrument. Cannot generalize...
    # TODO: What about call/put written on zero-coupon bond?
    #  Terminal condition should depend on zero-coupon bond, not option payoff
    solver.solution = instrument.payoff(solver.grid)

    return solver

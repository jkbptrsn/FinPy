import abc

import numpy as np
from scipy.linalg import solve_banded

from numerical_methods.finite_difference import misc
from utils import global_types
from utils import payoffs
from utils import timing


class Theta1D:
    """Theta method for solving parabolic 1-factor PDE (base class).

    The general structure of the PDE is
        dV/dt + drift * dV/dx + 1/2 * diffusion^2 * dV^2/dx^2 = rate * V,
    TODO: What about the RHS for Ornstein-Uhlenbeck processes?
    where the underlying 1-dimensional Markov process reads
        dx_t = drift(t, x_t) * dt + diffusion(t, x_t) * dW_t.

    The grid in the spatial dimension is assumed equidistant.

    Tri-diagonal form:
        - 1st row: Super-diagonal (not including first element)
        - 2nd row: Diagonal
        - 3rd row: Sub-diagonal (not including last element)

    TODO: Smoothing of payoff functions -- not necessary according to Andreasen
    TODO: Rannacher time stepping with fully implicit method -- not necessary according to Andreasen
    TODO: Upwinding -- rarely used by Andreasen
    """

    def __init__(self,
                 grid: np.ndarray,
                 form: str = "tri",
                 equidistant: bool = False):
        self.grid = grid
        self.form = form
        self.equidistant = equidistant

        self.vec_drift = None
        self.vec_diff_sq = None
        self.vec_rate = None
        self.vec_solution = None
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

    def set_drift(self, drift: np.ndarray):
        """Drift vector defined by underlying process."""
        self.vec_drift = drift

    def set_diffusion(self, diffusion: np.ndarray):
        """Squared diffusion vector defined by underlying process."""
        self.vec_diff_sq = np.square(diffusion)

    def set_rate(self, rate: np.ndarray):
        """Rate vector defined by underlying process."""
        self.vec_rate = rate

    @property
    def solution(self) -> np.ndarray:
        return self.vec_solution

    @solution.setter
    def solution(self, val: np.ndarray):
        self.vec_solution = val

    @abc.abstractmethod
    def set_propagator(self):
        pass

    @abc.abstractmethod
    def propagation(self, dt: float):
        pass

    def delta(self) -> np.ndarray:
        """Finite difference calculation of delta."""
        dx = self.grid[1] - self.grid[0]
        return misc.delta_equidistant(dx, self.vec_solution)

    def gamma(self) -> np.ndarray:
        """Finite difference calculation of gamma."""
        dx = self.grid[1] - self.grid[0]
        return misc.gamma_equidistant(dx, self.vec_solution)

    @abc.abstractmethod
    def theta(self, dt: float = None) -> np.ndarray:
        pass

    def theta_calc(self, dt: float) -> np.ndarray:
        """Theta calculated by central first order finite difference."""
        self.set_propagator()
        # Save current solution
        solution_copy = self.solution.copy()
        # Forward propagation
        dt = - dt
        self.propagation(dt)
        forward = self.solution.copy()
        # Backward propagation (two time steps)
        dt = - dt
        self.propagation(dt)
        self.propagation(dt)
        backward = self.solution.copy()
        # Restore current solution
        self.solution = solution_copy
        return (forward - backward) / (2 * dt)


class Andreasen1D(Theta1D):
    """The theta method implemented as shown in
    Jesper Andreasen's Finite Difference notes from 2011.

    The propagator is defined by
        dV/dt = - Propagator * V.

    The numerical solution is determined using
        - theta_parameter = 0   : Explicit method
        - theta_parameter = 1/2 : Crank-Nicolson method (default)
        - theta_parameter = 1   : Fully implicit method
    """

    def __init__(self,
                 grid: np.ndarray,
                 theta_parameter: float = 0.5):
        super().__init__(grid)
        self.theta_parameter = theta_parameter

        self.dt_last = None

    def initialization(self):
        """Initialization of identity matrix and propagator matrix."""
        self.mat_identity = misc.identity_matrix(self.nstates)
        self.set_propagator()

    def set_propagator(self):
        """Propagator on tri-diagonal form."""
        dx = self.grid[1] - self.grid[0]
        ddx = misc.ddx_equidistant(self.nstates, dx)
        d2dx2 = misc.d2dx2_equidistant(self.nstates, dx)
        self.mat_propagator = \
            - misc.dia_matrix_prod(self.vec_rate, self.mat_identity) \
            + misc.dia_matrix_prod(self.vec_drift, ddx) \
            + misc.dia_matrix_prod(self.vec_diff_sq, d2dx2) / 2

#    @timing.execution_time
    def propagation(self, dt: float):
        """Propagation of solution vector for one time step dt."""
        rhs = self.mat_identity \
            + (1 - self.theta_parameter) * dt * self.mat_propagator
        rhs = misc.matrix_col_prod(rhs, self.vec_solution)

        # Update propagator, if drift/diffusion are time-dependent.
        # But then one would also need to update vec_drift and vec_diff_sq...
#        self.set_propagator()

        lhs = self.mat_identity \
            - self.theta_parameter * dt * self.mat_propagator
        self.vec_solution = solve_banded((1, 1), lhs, rhs)
        self.dt_last = dt

    def theta(self, dt: float = None) -> np.ndarray:
        if not dt:
            if not self.dt_last:
                raise ValueError("dt should be set...")
            dt = self.dt_last
        return self.theta_calc(dt)


def norm_diff_1d(vec1: np.ndarray,
                 vec2: np.ndarray,
                 step_size1: float,
                 slice_nr=2):
    """

    Args:
        vec1:
        vec2:
        step_size1:
        slice_nr:

    Returns:

    """
    # Absolute difference. Exclude boundary points?

#    diff = np.abs(vec1[1:-1] - vec2[1:-1][::slice_nr])

#    if slice_nr == 1:
#        diff = np.abs(vec1[1:-1] - vec2[1:-1][::slice_nr])
#    elif slice_nr == 2:
#        diff = np.abs(vec1[1:-1] - vec2[2:-2][::slice_nr])

    diff = np.abs(vec1 - vec2[::slice_nr])

    # "Center" norm.
    n_states = diff.size
    idx_center = (n_states - 1) // 2
    norm_center = diff[idx_center]

    # Max norm.
    norm_max = np.amax(diff)

    # L2 norm.
    norm_l2 = np.sqrt(np.sum(np.square(diff)) * step_size1)

    return norm_center, norm_max, norm_l2


def setup_solver(instrument,
                 x_grid: np.ndarray,
                 theta: float = 0.5,
                 method: str = "Andreasen") \
        -> Andreasen1D:
    """Setting up finite difference solver.

    Args:
        instrument: Instrument object.
        x_grid: Grid in spatial dimension.
        theta: Theta parameter.
        method: "Andersen" og "Andreasen"

    Returns:
        Finite difference solver.
    """
    # Set up PDE solver.
    if method == "Andreasen":
        solver = Andreasen1D(x_grid, theta)
    else:
        raise ValueError("Method is not recognized.")
    if instrument.model == global_types.Model.BLACK_SCHOLES:
        drift = instrument.rate * solver.grid
        diffusion = instrument.vol * solver.grid
        rate = instrument.rate + 0 * solver.grid
    elif instrument.model == global_types.Model.BACHELIER:
        drift = 0 * solver.grid
        diffusion = instrument.vol + 0 * solver.grid
        rate = instrument.rate + 0 * solver.grid
    elif instrument.model == global_types.Model.CIR:
        drift = instrument.kappa * (instrument.mean_rate - solver.grid)
        diffusion = instrument.vol * np.sqrt(solver.grid)
        rate = solver.grid
    elif instrument.model == global_types.Model.VASICEK:
        drift = instrument.kappa * (instrument.mean_rate - solver.grid)
        diffusion = instrument.vol + 0 * solver.grid
        rate = solver.grid
    else:
        raise ValueError("Model is not recognized.")
    solver.set_drift(drift)
    solver.set_diffusion(diffusion)
    solver.set_rate(rate)

    # Terminal solution to PDE.
    if instrument.type == global_types.Instrument.EUROPEAN_CALL:
        solver.solution = payoffs.call(solver.grid, instrument.strike)
    elif instrument.type == global_types.Instrument.EUROPEAN_PUT:
        solver.solution = payoffs.put(solver.grid, instrument.strike)
    elif instrument.type == global_types.Instrument.ZERO_COUPON_BOND:
        solver.solution = payoffs.zero_coupon_bond(solver.grid)
    else:
        raise ValueError("Instrument is not recognized.")
    return solver

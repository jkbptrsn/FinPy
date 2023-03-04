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

    TODO: Already removed 2 extra grid points -- go through Andersen & Piterbarg
    TODO: Add non-equidistant grid. Instead of xmin, xmax, nstates, use state_grid as parameter
    TODO: Smoothing of payoff functions -- not necessary according to Andreasen
    TODO: Rannacher time stepping with fully implicit method -- not necessary according to Andreasen
    TODO: Upwinding -- rarely used by Andreasen
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int):
        self.xmin = xmin
        self.xmax = xmax

        # Adding boundary states.
        self.nstates = nstates
        self.dx = (xmax - xmin) / (nstates - 1)

        self.vec_drift = None
        self.vec_diff_sq = None
        self.vec_rate = None
        self.vec_solution = None
        self.mat_identity = None
        self.mat_propagator = None

    def grid(self) -> np.ndarray:
        """Equidistant grid between xmin and xmax including both points.
        Two boundary states are added at xmin - dx and xmax + dx.
        """
        return self.dx * np.arange(self.nstates) + self.xmin

    def set_drift(self, drift: np.ndarray):
        """Drift vector defined by the underlying stochastic process."""
        self.vec_drift = drift

    def set_diffusion(self, diffusion: np.ndarray):
        """Squared diffusion vector defined by the underlying stochastic
        process.
        """
        self.vec_diff_sq = np.square(diffusion)

    def set_rate(self, rate: np.ndarray):
        """Rate vector defined by the underlying stochastic process."""
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
    def propagation(self):
        pass

    def delta(self) -> np.ndarray:
        """Finite difference calculation of delta."""
        return misc.delta_equidistant(self.dx, self.vec_solution)

    def gamma(self) -> np.ndarray:
        """Finite difference calculation of gamma."""
        return misc.gamma_equidistant(self.dx, self.vec_solution)

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
        self.propagation()
        forward = self.solution.copy()
        # Backward propagation (two time steps)
        dt = - dt
        self.propagation()
        self.propagation()
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

    TODO: Remove dt as argument, give dt as argument in propagation function
    TODO: Give x_grid as argument
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 dt: float,
                 theta_parameter: float = 0.5):
        super().__init__(xmin, xmax, nstates)
        self.dt = dt
        self.theta_parameter = theta_parameter

    def initialization(self):
        """Initialization of identity matrix and propagator matrix."""
        self.mat_identity = misc.identity_matrix(self.nstates)
        self.set_propagator()

    def set_propagator(self):
        """Propagator on tri-diagonal form."""
        ddx = misc.ddx_equidistant(self.nstates, self.dx)
        d2dx2 = misc.d2dx2_equidistant(self.nstates, self.dx)
        self.mat_propagator = \
            - misc.dia_matrix_prod(self.vec_rate, self.mat_identity) \
            + misc.dia_matrix_prod(self.vec_drift, ddx) \
            + misc.dia_matrix_prod(self.vec_diff_sq, d2dx2) / 2

#    @timing.execution_time
    def propagation(self):
        """Propagation of solution vector for one time step dt."""
        rhs = self.mat_identity \
            + (1 - self.theta_parameter) * self.dt * self.mat_propagator
        rhs = misc.matrix_col_prod(rhs, self.vec_solution)

        # Update propagator, if drift/diffusion are time-dependent.
        # But then one would also need to update vec_drift and vec_diff_sq...
#        self.set_propagator()

        lhs = self.mat_identity \
            - self.theta_parameter * self.dt * self.mat_propagator
        self.vec_solution = solve_banded((1, 1), lhs, rhs)

    def theta(self, dt: float = None) -> np.ndarray:
        if not dt:
            dt = self.dt
        return self.theta_calc(dt)


def norm_diff_1d(vec1: np.ndarray,
                 vec2: np.ndarray,
                 step_size1: float,
                 slice_nr=2):

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
                 theta_value: float = 0.5,
                 method: str = "Andreasen") \
        -> Andreasen1D:
    """Setting up finite difference solver.

    Args:
        instrument: Instrument object.
        x_grid: Grid in spatial dimension.
        theta_value: ...
        method: "Andersen" og "Andreasen"

    Returns:
        Finite difference solver.
    """
    # Set up PDE solver.
    dt = instrument.event_grid[-1] - instrument.event_grid[-2]
    xmin = x_grid[0]
    xmax = x_grid[-1]
    nstates = x_grid.size
#    if method == "Andersen":
#        solver = Andersen1D(xmin, xmax, nstates, dt, theta_value)
    if method == "Andreasen":
        solver = Andreasen1D(xmin, xmax, nstates, dt, theta_value)
    else:
        raise ValueError("Method is not recognized.")
    if instrument.model == global_types.Model.BLACK_SCHOLES:
        drift = instrument.rate * solver.grid()
        diffusion = instrument.vol * solver.grid()
        rate = instrument.rate + 0 * solver.grid()
    elif instrument.model == global_types.Model.BACHELIER:
        drift = 0 * solver.grid()
        diffusion = instrument.vol + 0 * solver.grid()
        rate = instrument.rate + 0 * solver.grid()
    elif instrument.model == global_types.Model.CIR:
        drift = instrument.kappa * (instrument.mean_rate - solver.grid())
        diffusion = instrument.vol * np.sqrt(solver.grid())
        rate = solver.grid()
    elif instrument.model == global_types.Model.VASICEK:
        drift = instrument.kappa * (instrument.mean_rate - solver.grid())
        diffusion = instrument.vol + 0 * solver.grid()
        rate = solver.grid()
    else:
        raise ValueError("Model is not recognized.")
    solver.set_drift(drift)
    solver.set_diffusion(diffusion)
    solver.set_rate(rate)

    # Terminal solution to PDE.
    if instrument.type == global_types.Instrument.EUROPEAN_CALL:
        solver.solution = payoffs.call(solver.grid(), instrument.strike)
    elif instrument.type == global_types.Instrument.EUROPEAN_PUT:
        solver.solution = payoffs.put(solver.grid(), instrument.strike)
    elif instrument.type == global_types.Instrument.ZERO_COUPON_BOND:
        solver.solution = payoffs.zero_coupon_bond(solver.grid())
    else:
        raise ValueError("Instrument is not recognized.")
    return solver

import abc

import numpy as np
from scipy.linalg import solve_banded

from utils import timing
from utils import global_types
from utils import payoffs


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

    TODO: Remove 2 extra grid points -- need for convergence tests? Go through Andersen & Piterbarg
    TODO: Add non-equidistant grid. Instead of xmin, xmax, nstates, use state_grid as parameter
    TODO: Smoothing of payoff functions -- not necessary according to Andreasen
    TODO: Rannacher time stepping with fully implicit method -- not necessary according to Andreasen
    TODO: Upwinding -- rarely used by Andreasen
    TODO: From tri to penta?
    TODO: Static methods in separate file, with identity matrix and matrix products for both tri and penta
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int):
        self.xmin = xmin
        self.xmax = xmax

        # Adding boundary states.
        self.nstates = nstates + 2
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
        return self.dx * np.arange(-1, self.nstates - 1) + self.xmin

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
        """Delta calculated by second order finite differences.

        Assuming equidistant and ascending grid.
        """
        delta = np.zeros(self.nstates)
        # Central finite difference.
        delta[1:-1] = (self.vec_solution[2:]
                       - self.vec_solution[:-2]) / (2 * self.dx)
        # Forward finite difference.
        delta[0] = (- self.vec_solution[2] / 2
                    + 2 * self.vec_solution[1]
                    - 3 * self.vec_solution[0] / 2) / self.dx
        # Backward finite difference.
        delta[-1] = (self.vec_solution[-3] / 2
                     - 2 * self.vec_solution[-2]
                     + 3 * self.vec_solution[-1] / 2) / self.dx
        return delta

    def gamma(self) -> np.ndarray:
        """Gamma calculated by second order finite differences.

        Assuming equidistant and ascending grid.
        """
        dx_sq = self.dx ** 2
        gamma = np.zeros(self.nstates)
        # Central finite difference
        gamma[1:-1] = (self.vec_solution[2:]
                       + self.vec_solution[:-2]
                       - 2 * self.vec_solution[1:-1]) / dx_sq
        # Forward finite difference
        gamma[0] = (- self.vec_solution[3]
                    + 4 * self.vec_solution[2]
                    - 5 * self.vec_solution[1]
                    + 2 * self.vec_solution[0]) / dx_sq
        # Backward finite difference
        gamma[-1] = (- self.vec_solution[-4]
                     + 4 * self.vec_solution[-3]
                     - 5 * self.vec_solution[-2]
                     + 2 * self.vec_solution[-1]) / dx_sq
        return gamma

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

    @staticmethod
    def identity_matrix(n_elements) -> np.ndarray:
        """Identity matrix on tri-diagonal form."""
        mat = np.zeros((3, n_elements))
        mat[1, :] = 1
        return mat

    @staticmethod
    def matrix_col_prod(matrix: np.ndarray,
                        vector: np.ndarray) -> np.ndarray:
        """Product of tri-diagonal matrix and column vector."""
        # Contribution from diagonal.
        product = matrix[1, :] * vector
        # Contribution from super-diagonal.
        product[:-1] += matrix[0, 1:] * vector[1:]
        # Contribution from sub-diagonal.
        product[1:] += matrix[2, :-1] * vector[:-1]
        return product

    @staticmethod
    def row_matrix_prod(vector: np.ndarray,
                        matrix: np.ndarray) -> np.ndarray:
        """Product of row vector and tri-diagonal matrix."""
        product = np.zeros(matrix.shape)
        # Contribution from super-diagonal.
        product[0, 1:] = vector[:-1] * matrix[0, 1:]
        # Contribution from diagonal.
        product[1, :] = vector * matrix[1, :]
        # Contribution from sub-diagonal.
        product[2, :-1] = vector[1:] * matrix[2, :-1]
        return product


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
    TODO: Move ddx and d2dx2 to separate file, generalize for penta...
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

    def ddx(self) -> np.ndarray:
        """Central finite difference approximation of first order
        derivative operator. At the boundaries, first order
        forward/backward difference is used.
        """
        matrix = np.zeros((3, self.nstates))
        # Central difference.
        matrix[0, 2:] = 1
        matrix[2, :-2] = -1
        # Forward difference at lower boundary.
        matrix[0, 1] = 2
        matrix[1, 0] = -2
        # Backward difference at upper boundary.
        matrix[1, -1] = 2
        matrix[2, -2] = -2
        return matrix / (2 * self.dx)

    def d2dx2(self) -> np.ndarray:
        """Central finite difference approximation of second order
        derivative operator. At the boundaries, the operator is set
        equal to zero.
        """
        matrix = np.zeros((3, self.nstates))
        matrix[0, 2:] = 1
        matrix[1, 1:-1] = -2
        matrix[2, :-2] = 1
        return matrix / (self.dx ** 2)

    def initialization(self):
        """Initialization of identity matrix and propagator matrix."""
        self.mat_identity = self.identity_matrix(self.nstates)
        self.set_propagator()

    def set_propagator(self):
        """Propagator on tri-diagonal form."""
        self.mat_propagator = \
            - self.row_matrix_prod(self.vec_rate, self.mat_identity) \
            + self.row_matrix_prod(self.vec_drift, self.ddx()) \
            + self.row_matrix_prod(self.vec_diff_sq, self.d2dx2()) / 2

#    @timing.execution_time
    def propagation(self):
        """Propagation of solution vector for one time step dt."""
        rhs = self.mat_identity \
            + (1 - self.theta_parameter) * self.dt * self.mat_propagator
        rhs = self.matrix_col_prod(rhs, self.vec_solution)

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


class Andersen1D(Theta1D):
    """The theta method implemented as shown in
    L.B.G. Andersen & V.V. Piterbarg 2010.

    TODO: Give x_grid as argument
    TODO: "PDE" boundary conditions requires a very small initial time step,
        and the convergence order differs from "Linearity". Hence, use
        "Linearity" boundary conditions in "production".
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 dt: float,
                 theta_parameter: float = 0.5,
                 bc_type: str = "Linearity"):
        super().__init__(xmin, xmax, nstates)
        self.dt = dt
        self.theta_parameter = theta_parameter
        self.bc_type = bc_type
        self.vec_boundary = None
        self.bc = np.zeros(2)

    def initialization(self):
        """Initialization of identity matrix, boundary conditions and
        propagator matrix.
        """
        self.mat_identity = self.identity_matrix(self.nstates - 2)
        self.mat_propagator = np.zeros((3, self.nstates - 2))
        self.vec_boundary = np.zeros(self.nstates - 2)
        # Save boundary conditions at previous time step.
        self.set_boundary_conditions_dt()
        self.set_propagator()

    def set_propagator(self):
        """Propagator on tri-diagonal form."""
        dx_sq = self.dx ** 2
        # Eq. (2.7) - (2.9), L.B.G. Andersen & V.V. Piterbarg 2010.
        upper = (self.vec_drift / self.dx + self.vec_diff_sq / dx_sq) / 2
        center = - self.vec_diff_sq / dx_sq - self.vec_rate
        lower = (-self.vec_drift / self.dx + self.vec_diff_sq / dx_sq) / 2
        # Keep elements for interior states.
        upper = upper[1:-1]
        center = center[1:-1]
        lower = lower[1:-1]
        # Set propagator matrix consistent with the solve_banded
        # function (scipy.linalg)
        # Eq. (2.11), L.B.G. Andersen & V.V. Piterbarg 2010.
        self.mat_propagator[0, 1:] = upper[:-1]
        self.mat_propagator[1, :] = center
        self.mat_propagator[2, :-1] = lower[1:]
        # Boundary conditions.
        k1, k2, km_1, km, f1, fm = self.boundary_conditions()
        # Adjust propagator matrix for boundary conditions
        # Eq. (2.12) - (2.13), L.B.G. Andersen & V.V. Piterbarg 2010.
        self.mat_propagator[1, -1] += km * upper[-1]
        self.mat_propagator[2, -2] += km_1 * upper[-1]
        self.mat_propagator[1, 0] += k1 * lower[0]
        self.mat_propagator[0, 1] += k2 * lower[0]
        # Set boundary vector
        self.vec_boundary[0] = lower[0] * f1
        self.vec_boundary[-1] = upper[-1] * fm

#    @timing.execution_time
    def propagation(self):
        """Propagation of solution vector for one time step dt."""
        # Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010.
        rhs = self.mat_identity \
            + (1 - self.theta_parameter) * self.dt * self.mat_propagator
        rhs = self.matrix_col_prod(rhs, self.vec_solution[1:-1]) \
            + (1 - self.theta_parameter) * self.dt * self.vec_boundary
        # Save boundary conditions at previous time step.
        self.set_boundary_conditions_dt()
        # Update self.mat_propagator and self.vec_boundary at t - dt.
        self.set_propagator()
        # Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010.
        rhs += self.theta_parameter * self.dt * self.vec_boundary
        lhs = self.mat_identity \
            - self.theta_parameter * self.dt * self.mat_propagator
        # Solve Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010.
        self.vec_solution[1:-1] = solve_banded((1, 1), lhs, rhs)
        # Boundary conditions.
        k1, k2, km_1, km, f1, fm = self.boundary_conditions()
        # Eq. (2.12) - (2.13), L.B.G. Andersen & V.V. Piterbarg 2010.
        self.vec_solution[0] = \
            k1 * self.vec_solution[1] + k2 * self.vec_solution[2] + f1
        self.vec_solution[-1] = \
            km * self.vec_solution[-2] + km_1 * self.vec_solution[-3] + fm

    def boundary_conditions(self):
        """Calculate coefficients in Eq. (2.12) - (2.13),
        L.B.G. Andersen & V.V. Piterbarg 2010, based on boundary
        conditions:
            - k1   = k_1(t)
            - k2   = k_2(t)
            - km_1 = k_{m - 1}(t)
            - km   = k_m(t)
            - f1   = underline{f}(t)
            - fm   = overline{f}(t)
        Also, see notes.
        """
        if self.bc_type == "Linearity":
            return 2, -1, -1, 2, 0, 0
        elif self.bc_type == "PDE":
            a, b, c, a_p, b_p, c_p = self.bc_coefficients()
            k1 = - b / a
            k2 = - c / a
            f1 = self.bc[0] / a
            km = - b_p / a_p
            km_1 = - c_p / a_p
            fm = self.bc[-1] / a_p
            return k1, k2, km_1, km, f1, fm
        else:
            raise ValueError(f"_bc_type can be either "
                             f"\"Linearity\" or \"PDE\": {self.bc_type}")

    def bc_coefficients(self) -> tuple:
        """Coefficients at time t, see notes.
        Also, section 10.1.5.2, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dx_sq = self.dx ** 2
        theta_dt = self.theta_parameter * self.dt
        # Lower spatial boundary.
        a = 1 + theta_dt * self.vec_drift[0] / self.dx \
            - theta_dt * self.vec_diff_sq[0] / (2 * dx_sq) \
            + theta_dt * self.vec_rate[0]
        b = theta_dt * self.vec_diff_sq[0] / dx_sq \
            - theta_dt * self.vec_drift[0] / self.dx
        c = - theta_dt * self.vec_diff_sq[0] / (2 * dx_sq)
        # Upper spatial boundary.
        a_p = 1 - theta_dt * self.vec_drift[-1] / self.dx \
            - theta_dt * self.vec_diff_sq[-1] / (2 * dx_sq) \
            + theta_dt * self.vec_rate[-1]
        b_p = theta_dt * self.vec_diff_sq[-1] / dx_sq \
            + theta_dt * self.vec_drift[-1] / self.dx
        c_p = - theta_dt * self.vec_diff_sq[-1] / (2 * dx_sq)
        return a, b, c, a_p, b_p, c_p

    def bc_coefficients_dt(self) -> tuple:
        """Coefficients at time t + dt, see notes.
        Also, section 10.1.5.2, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        dx_sq = self.dx ** 2
        theta_dt = (1 - self.theta_parameter) * self.dt
        # Lower spatial boundary.
        d = 1 - theta_dt * self.vec_drift[0] / self.dx \
            + theta_dt * self.vec_diff_sq[0] / (2 * dx_sq) \
            - theta_dt * self.vec_rate[0]
        e = - theta_dt * self.vec_diff_sq[0] / dx_sq \
            + theta_dt * self.vec_drift[0] / self.dx
        f = theta_dt * self.vec_diff_sq[0] / (2 * dx_sq)
        # Upper spatial boundary.
        d_p = 1 + theta_dt * self.vec_drift[-1] / self.dx \
            + theta_dt * self.vec_diff_sq[-1] / (2 * dx_sq) \
            - theta_dt * self.vec_rate[-1]
        e_p = - theta_dt * self.vec_diff_sq[-1] / dx_sq \
            - theta_dt * self.vec_drift[-1] / self.dx
        f_p = theta_dt * self.vec_diff_sq[-1] / (2 * dx_sq)
        return d, e, f, d_p, e_p, f_p

    def set_boundary_conditions_dt(self):
        """Coefficients at time t + dt, see notes.
        Also, section 10.1.5.2, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        d, e, f, d_p, e_p, f_p = self.bc_coefficients_dt()
        # Lower spatial boundary.
        self.bc[0] = \
            d * self.vec_solution[0] \
            + e * self.vec_solution[1] \
            + f * self.vec_solution[2]
        # Upper spatial boundary.
        self.bc[-1] = \
            d_p * self.vec_solution[-1] \
            + e_p * self.vec_solution[-2] \
            + f_p * self.vec_solution[-3]

    def theta(self, dt: float = None) -> np.ndarray:
        if not dt:
            dt = self.dt
        return self.theta_calc(dt)


def setup_black_scholes(xmin: float,
                        xmax: float,
                        nstates: int,
                        dt: float,
                        rate: float,
                        vol: float,
                        theta: float = 0.5,
                        method: str = "Andreasen"):
    """Set up Black-Scholes PDE...

    TODO: Remove and replace by setup_solver
    """
    # Set up PDE solver.
    if method == "Andersen":
        solver = Andersen1D(xmin, xmax, nstates, dt, theta)
    elif method == "Andreasen":
        solver = Andreasen1D(xmin, xmax, nstates, dt, theta)
    else:
        raise ValueError("Method is not recognized.")
    # Black-Scholes PDE.
    solver.set_drift(rate * solver.grid())
    solver.set_diffusion(vol * solver.grid())
    solver.set_rate(rate + 0 * solver.grid())
    return solver


def setup_vasicek(xmin: float,
                  xmax: float,
                  nstates: int,
                  dt: float,
                  kappa: float,
                  mean_rate: float,
                  vol: float,
                  theta: float = 0.5,
                  method: str = "Andreasen"):
    """Set up Vasicek PDE...

    TODO: Remove and replace by setup_solver
    """
    # Set up PDE solver.
    if method == "Andersen":
        solver = Andersen1D(xmin, xmax, nstates, dt, theta)
    elif method == "Andreasen":
        solver = Andreasen1D(xmin, xmax, nstates, dt, theta)
    else:
        raise ValueError("Method is not recognized.")
    # Vasicek PDE.
    solver.set_drift(kappa * (mean_rate - solver.grid()))
    solver.set_diffusion(vol + 0 * solver.grid())
    solver.set_rate(solver.grid())
    return solver


def norm_diff_1d(vec1: np.ndarray,
                 vec2: np.ndarray,
                 step_size1: float,
                 slice_nr=2):

    # Absolute difference. Exclude boundary points?
    diff = np.abs(vec1[1:-1] - vec2[1:-1][::slice_nr])

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
                 method: str = "Andersen") \
        -> (Andersen1D, Andreasen1D):
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
    if method == "Andersen":
        solver = Andersen1D(xmin, xmax, nstates, dt, theta_value)
    elif method == "Andreasen":
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

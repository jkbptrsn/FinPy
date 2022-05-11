import abc
import numpy as np
from scipy.linalg import solve_banded


class Theta:
    """Base class: Theta scheme for solving a parabolic 1-factor PDE.

    The general structure of the PDE is
    dV/dt + drift * dV/dx + 1/2 * diffusion^2 * dV^2/dx^2 = rate * V

    The numerical solution is determined using the theta method:
        - theta = 0   : Explicit Euler
        - theta = 1/2 : Crank-Nicolson (default)
        - theta = 1   : Fully implicit

    The grid in the spatial dimension is assumed equidistant.

    TODO: Add non-equidistant grid
    TODO: Smoothing of payoff functions -- not necessary according to Andreasen
    TODO: Rannacher time stepping with fully implicit method -- not necessary according to Andreasen
    TODO: Upwinding
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 dt: float,
                 theta: float = 0.5):
        self._xmin = xmin
        self._xmax = xmax
        # Adding boundary states
        self._nstates = nstates + 2
        self._dt = dt
        self._theta = theta
        self._dx = (xmax - xmin) / (nstates - 1)
        self._vec_drift = None
        self._vec_diff_sq = None
        self._vec_rate = None
        self._vec_solution = None
        self._mat_identity = None
        self._mat_propagator = None

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def nstates(self) -> int:
        # Removing boundary states
        return self._nstates - 2

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, val: float):
        self._dt = val

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, val: float):
        self._theta = val

    @property
    def dx(self) -> float:
        return self._dx

    def grid(self) -> np.ndarray:
        """Equidistant grid between _xmin and _xmax including both
        points. Two boundary states are added at _xmin - _dx and
        _xmax + _dx.
        """
        return self._dx * np.arange(-1, self._nstates - 1) + self._xmin

    def set_drift(self, drift: np.ndarray):
        """Drift vector defined by the underlying stochastic process."""
        self._vec_drift = drift

    def set_diffusion(self, diffusion: np.ndarray):
        """Squared diffusion vector defined by the underlying stochastic
        process.
        """
        self._vec_diff_sq = np.square(diffusion)

    def set_rate(self, rate: np.ndarray):
        """Rate vector defined by the underlying stochastic process."""
        self._vec_rate = rate

    @property
    def solution(self) -> np.ndarray:
        return self._vec_solution

    @solution.setter
    def solution(self, val: np.ndarray):
        self._vec_solution = val

    @abc.abstractmethod
    def set_propagator(self):
        pass

    @abc.abstractmethod
    def propagation(self):
        pass

    def delta_fd(self) -> np.ndarray:
        """Delta calculated by second order finite differences. Assuming
        equidistant and ascending grid.
        """
        delta = np.zeros(self._nstates)
        # Central finite difference
        delta[1:-1] = (self._vec_solution[2:]
                       - self._vec_solution[:-2]) / (2 * self._dx)
        # Forward finite difference
        delta[0] = (- self._vec_solution[2] / 2
                    + 2 * self._vec_solution[1]
                    - 3 * self._vec_solution[0] / 2) / self._dx
        # Backward finite difference
        delta[-1] = (self._vec_solution[-3] / 2
                     - 2 * self._vec_solution[-2]
                     + 3 * self._vec_solution[-1] / 2) / self._dx
        return delta

    def gamma_fd(self) -> np.ndarray:
        """Gamma calculated by second order finite differences. Assuming
        equidistant and ascending grid.
        """
        dx_sq = self._dx ** 2
        gamma = np.zeros(self._nstates)
        # Central finite difference
        gamma[1:-1] = (self._vec_solution[2:]
                       + self._vec_solution[:-2]
                       - 2 * self._vec_solution[1:-1]) / dx_sq
        # Forward finite difference
        gamma[0] = (- self._vec_solution[3]
                    + 4 * self._vec_solution[2]
                    - 5 * self._vec_solution[1]
                    + 2 * self._vec_solution[0]) / dx_sq
        # Backward finite difference
        gamma[-1] = (- self._vec_solution[-4]
                     + 4 * self._vec_solution[-3]
                     - 5 * self._vec_solution[-2]
                     + 2 * self._vec_solution[-1]) / dx_sq
        return gamma

    def theta_fd(self) -> np.ndarray:
        """Theta calculated by central finite difference.

        TODO: Check AndersenPiterbarg1D with boundary = "PDE"?
        """
        self.set_propagator()
        # Save current solution
        solution_copy = self.solution.copy()
        # Forward propagation
        self._dt = - self._dt
        self.propagation()
        forward = self.solution.copy()
        # Backward propagation (two time steps)
        self._dt = - self._dt
        self.propagation()
        self.propagation()
        backward = self.solution.copy()
        # Restore current solution
        self.solution = solution_copy
        return (forward - backward) / (2 * self._dt)

    @staticmethod
    def identity_matrix(n_elements):
        """Identity matrix on tri-diagonal form.
            - 1st row: Super-diagonal (not including first element)
            - 2nd row: Diagonal
            - 3rd row: Sub-diagonal (not including last element)
        """
        mat_identity = np.zeros((3, n_elements))
        mat_identity[1, :] = 1
        return mat_identity

    @staticmethod
    def mat_vec_product(matrix: np.ndarray,
                        vector: np.ndarray) -> np.ndarray:
        """Product of tri-diagonal matrix and column vector."""
        # Contribution from diagonal
        product = matrix[1, :] * vector
        # Contribution from super-diagonal
        product[:-1] += matrix[0, 1:] * vector[1:]
        # Contribution from sub-diagonal
        product[1:] += matrix[2, :-1] * vector[:-1]
        return product


class Andreasen1D(Theta):
    """The theta method implemented as shown in
    Jesper Andreasen's PDE notes from 2011.

    TODO: Check convergence -- should match AndersenPiterbarg1D using bc_type = "Linearity"
    TODO: Doesn't work correctly for Vasicek short rate model
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 dt: float,
                 theta: float = 0.5):
        super().__init__(xmin, xmax, nstates, dt, theta=theta)

    def mat_product(self,
                    diagonal: np.ndarray,
                    tridiagonal: np.ndarray) -> np.ndarray:
        """Product of diagonal and tri-diagonal matrices."""
        matrix = np.zeros((3, self._nstates))
        matrix[0, 1:] = diagonal[:-1] * tridiagonal[0, 1:]
        matrix[1, :] = diagonal * tridiagonal[1, :]
        matrix[2, :-1] = diagonal[1:] * tridiagonal[2, :-1]
        return matrix

    def ddx(self) -> np.ndarray:
        """Central finite difference approximation of first order
        derivative operator. At the boundaries, first order
        forward/backward difference is used.
        """
        matrix = np.zeros((3, self._nstates))
        # Central difference
        matrix[0, 2:] = 1
        matrix[2, :-2] = -1
        # Forward difference at lower boundary
        matrix[0, 1] = 2
        matrix[1, 0] = -2
        # Backward difference at upper boundary
        matrix[1, -1] = 2
        matrix[2, -2] = -2
        return matrix / (2 * self._dx)

    def d2dx2(self) -> np.ndarray:
        """Central finite difference approximation of second order
        derivative operator. At the boundaries, the operator is set
        equal to zero.
        """
        matrix = np.zeros((3, self._nstates))
        matrix[0, 2:] = 1
        matrix[1, 1:-1] = -2
        matrix[2, :-2] = 1
        return matrix / self._dx ** 2

    def initialization(self):
        """Initialization of identity matrix, boundary conditions and
        propagator matrix.
        """
        self._mat_identity = self.identity_matrix(self._nstates)
        self.set_propagator()

    def set_propagator(self):
        """Propagator on tri-diagonal form.
            - 1st row: Super-diagonal (not including first element)
            - 2nd row: Diagonal
            - 3rd row: Sub-diagonal (not including last element)
        """
        self._mat_propagator = \
            self.mat_product(self._vec_drift, self.ddx()) \
            + self.mat_product(self._vec_diff_sq, self.d2dx2()) / 2 \
            - self._vec_rate

    def propagation(self):
        """Propagation of solution vector for one time step _dt."""
        rhs = self._mat_identity \
            + (1 - self._theta) * self._dt * self._mat_propagator
        rhs = self.mat_vec_product(rhs, self._vec_solution)
        self.set_propagator()
        lhs = self._mat_identity \
            - self._theta * self._dt * self._mat_propagator
        self._vec_solution = solve_banded((1, 1), lhs, rhs)


class AndersenPiterbarg1D(Theta):
    """The theta method implemented as shown in
    L.B.G. Andersen & V.V. Piterbarg 2010.
    """

    def __init__(self,
                 xmin: float,
                 xmax: float,
                 nstates: int,
                 dt: float,
                 theta: float = 0.5,
                 bc_type: str = "Linearity"):
        super().__init__(xmin, xmax, nstates, dt, theta=theta)

        self._bc_type = bc_type
        self._vec_boundary = None
        self._bc = np.zeros(2)

    @property
    def bc_type(self) -> str:
        return self._bc_type

    def initialization(self):
        """Initialization of identity matrix, boundary conditions and
        propagator matrix.
        """
        self._mat_identity = self.identity_matrix(self._nstates - 2)
        self.set_boundary_conditions_dt()
        self.set_propagator()

    def set_propagator(self):
        """Propagator on tri-diagonal form.
            - 1st row: Super-diagonal (not including first element)
            - 2nd row: Diagonal
            - 3rd row: Sub-diagonal (not including last element)
        """
        dx_sq = self._dx ** 2
        # Eq. (2.7) - (2.9), L.B.G. Andersen & V.V. Piterbarg 2010
        upper = (self._vec_drift / self._dx + self._vec_diff_sq / dx_sq) / 2
        center = - self._vec_diff_sq / dx_sq - self._vec_rate
        lower = (-self._vec_drift / self._dx + self._vec_diff_sq / dx_sq) / 2
        # Keep elements for interior states
        upper = upper[1:-1]
        center = center[1:-1]
        lower = lower[1:-1]
        # Set propagator matrix consistent with the solve_banded
        # function (scipy.linalg)
        # Eq. (2.11), L.B.G. Andersen & V.V. Piterbarg 2010
        self._mat_propagator = np.zeros((3, self._nstates - 2))
        self._mat_propagator[0, 1:] = upper[:-1]
        self._mat_propagator[1, :] = center
        self._mat_propagator[2, :-1] = lower[1:]
        # Boundary conditions
        k1, k2, km_1, km, f1, fm = self.boundary_conditions()
        # Adjust propagator matrix for boundary conditions
        # Eq. (2.12) - (2.13), L.B.G. Andersen & V.V. Piterbarg 2010
        self._mat_propagator[1, -1] += km * upper[-1]
        self._mat_propagator[2, -2] += km_1 * upper[-1]
        self._mat_propagator[1, 0] += k1 * lower[0]
        self._mat_propagator[0, 1] += k2 * lower[0]
        # Set boundary vector
        self._vec_boundary = np.zeros(self._nstates - 2)
        self._vec_boundary[0] = lower[0] * f1
        self._vec_boundary[-1] = upper[-1] * fm

    def propagation(self):
        """Propagation of solution vector for one time step _dt."""
        # Save boundary conditions at previous time step
        self.set_boundary_conditions_dt()
        # Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010
        rhs = self._mat_identity \
            + (1 - self.theta) * self._dt * self._mat_propagator
        rhs = self.mat_vec_product(rhs, self._vec_solution[1:-1]) \
            + (1 - self._theta) * self._dt * self._vec_boundary
        # Update self._mat_propagator and self._vec_boundary
        self.set_propagator()
        # Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010
        rhs += self._theta * self._dt * self._vec_boundary
        lhs = self._mat_identity - self.theta * self._dt * self._mat_propagator
        # Solve Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010
        self._vec_solution[1:-1] = solve_banded((1, 1), lhs, rhs)
        # Boundary conditions
        k1, k2, km_1, km, f1, fm = self.boundary_conditions()
        # Eq. (2.12) - (2.13), L.B.G. Andersen & V.V. Piterbarg 2010
        self._vec_solution[0] = \
            k1 * self._vec_solution[1] + k2 * self._vec_solution[2] + f1
        self._vec_solution[-1] = \
            km * self._vec_solution[-2] + km_1 * self._vec_solution[-3] + fm

    def boundary_conditions(self):
        """Calculate coefficients in Eq. (2.12) - (2.13),
        L.B.G. Andersen & V.V. Piterbarg 2010, based on boundary
        conditions:
            - k1   = k_1(t)
            - k2   = k_2(t)
            - km_1 = k_{m - 1}(t)
            - km   = k_m(t)
            - f1   = overline{f}(t)
            - fm   = underline{f}(t)
        """
        if self._bc_type == "Linearity":
            return 2, -1, -1, 2, 0, 0
        elif self._bc_type == "PDE":
            a, b, c, a_p, b_p, c_p = self.bc_coefficients()
            k1 = - b / a
            k2 = - c / a
            f1 = self._bc[0] / a
            km = - b_p / a_p
            km_1 = - c_p / a_p
            fm = self._bc[-1] / a_p
            return k1, k2, km_1, km, f1, fm
        else:
            raise ValueError(f"_bc_type can be either "
                             f"\"Linearity\" or \"PDE\": {self._bc_type}")

    def bc_coefficients(self) -> tuple:
        """Section 10.1.5.2, L.B.G. Andersen & V.V. Piterbarg 2010."""
        dx_sq = self._dx ** 2
        theta_dt = self._theta * self._dt
        # Lower boundary
        a = 1 + theta_dt * self._vec_drift[0] / self._dx \
            - theta_dt * self._vec_diff_sq[0] / (2 * dx_sq) \
            + theta_dt * self._vec_rate[0]
        b = theta_dt * self._vec_diff_sq[0] / dx_sq \
            - theta_dt * self._vec_drift[0] / self._dx
        c = - theta_dt * self._vec_diff_sq[0] / (2 * dx_sq)
        # Upper boundary
        a_p = 1 - theta_dt * self._vec_drift[-1] / self._dx \
            - theta_dt * self._vec_diff_sq[-1] / (2 * dx_sq) \
            + theta_dt * self._vec_rate[-1]
        b_p = theta_dt * self._vec_diff_sq[-1] / dx_sq \
            + theta_dt * self._vec_drift[-1] / self._dx
        c_p = - theta_dt * self._vec_diff_sq[-1] / (2 * dx_sq)
        return a, b, c, a_p, b_p, c_p

    def bc_coefficients_dt(self) -> tuple:
        """Section 10.1.5.2, L.B.G. Andersen & V.V. Piterbarg 2010."""
        dx_sq = self._dx ** 2
        theta_dt = (1 - self._theta) * self._dt
        # Lower boundary
        d = 1 - theta_dt * self._vec_drift[0] / self._dx \
            + theta_dt * self._vec_diff_sq[0] / (2 * dx_sq) \
            - theta_dt * self._vec_rate[0]
        e = - theta_dt * self._vec_diff_sq[0] / dx_sq \
            + theta_dt * self._vec_drift[0] / self._dx
        f = theta_dt * self._vec_diff_sq[0] / (2 * dx_sq)
        # Upper boundary
        d_p = 1 + theta_dt * self._vec_drift[-1] / self._dx \
            + theta_dt * self._vec_diff_sq[-1] / (2 * dx_sq) \
            - theta_dt * self._vec_rate[-1]
        e_p = - theta_dt * self._vec_diff_sq[-1] / dx_sq \
            - theta_dt * self._vec_drift[-1] / self._dx
        f_p = theta_dt * self._vec_diff_sq[-1] / (2 * dx_sq)
        return d, e, f, d_p, e_p, f_p

    def set_boundary_conditions_dt(self):
        """Section 10.1.5.2, L.B.G. Andersen & V.V. Piterbarg 2010."""
        d, e, f, d_p, e_p, f_p = self.bc_coefficients_dt()
        # Lower boundary
        self._bc[0] = \
            d * self._vec_solution[0] \
            + e * self._vec_solution[1] \
            + f * self._vec_solution[2]
        # Upper boundary
        self._bc[-1] = \
            d_p * self._vec_solution[-1] \
            + e_p * self._vec_solution[-2] \
            + f_p * self._vec_solution[-3]

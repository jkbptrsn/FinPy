import numpy as np
from scipy.linalg import solve_banded


class SolverOld:
    """Solve parabolic 1-d PDEs using the theta method:
    - theta = 0: Explicit method
    - theta = 1/2: Crank-Nicolson method (default)
    - theta = 1: Fully implicit method"""

    # todo: Choose boundary conditions

    def __init__(self, xmin, xmax, n, theta=0.5):
        self._xmin = xmin
        self._xmax = xmax
        self._n = n
        self._theta = theta
        self._dx = (xmax - xmin) / (n - 1)

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def n(self):
        return self._n

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, val):
        self._theta = val

    @property
    def dx(self):
        return self._dx

    def x_grid(self):
        return self.dx * np.array(range(self.n)) + self.xmin

    def identity(self,
                 vector: np.ndarray = None) -> np.ndarray:
        """Identity matrix on tri-diagonal form:
        - 1st row: Super-diagonal
        - 2nd row: Diagonal
        - 3rd row: Sub-diagonal
        Construct diagonal element-wise from vector."""
        matrix = np.zeros((3, self.n))
        matrix[1, :] = 1
        if vector is None:
            return matrix
        else:
            matrix[1, :] = vector
            return matrix

    def diag_tridiag(self,
                     diagonal: np.ndarray,
                     tridiagonal: np.ndarray) -> np.ndarray:
        """Product of diagonal and tri-diagonal matrices, both in
        tri-diagonal form."""
        product = np.zeros((3, self.n))
        product[0, 1:] = diagonal[1, :-1] * tridiagonal[0, 1:]
        product[1, :] = diagonal[1, :] * tridiagonal[1, :]
        product[2, :-1] = diagonal[1, 1:] * tridiagonal[2, :-1]
        return product

    def tridiag_vec(self,
                    tridiagonal: np.ndarray,
                    vector: np.ndarray) -> np.ndarray:
        """Product of tri-diagonal matrix and column vector."""
        # Contribution from diagonal
        product = tridiagonal[1, :] * vector
        # Contribution from super-diagonal
        product[:-1] += tridiagonal[0, 1:] * vector[1:]
        # Contribution from sub-diagonal
        product[1:] += tridiagonal[2, :-1] * vector[:-1]
        return product

    def ddx(self) -> np.ndarray:
        """Central difference approximation of 1st order derivative
        operator. At the boundaries, forward/backward difference is
        used."""
        matrix = np.zeros((3, self.n))
        # Central difference
        matrix[0, 2:] = 1
        matrix[2, :-2] = -1
        # Forward difference at xmin
        matrix[0, 1] = 2
        matrix[1, 0] = -2
        # Backward difference at xmax
        matrix[1, -1] = 2
        matrix[2, -2] = -2
        return matrix / (2 * self.dx)

    def vector_ddx(self,
                   vector: np.ndarray) -> np.ndarray:
        """Product of input vector and 1st order derivative operator."""
        return self.diag_tridiag(self.identity(vector), self.ddx())

    def x_ddx(self) -> np.ndarray:
        """Product of space coordinate and its 1st order derivative
        operator."""
#        return self.diag_tridiag(self.identity(self.x_grid()), self.ddx())
        return self.vector_ddx(self.x_grid())

    def d2dx2(self) -> np.ndarray:
        """Finite difference approximation of 2nd order derivative
        operator. At the boundaries, the operator is set equal to
        zero."""
        matrix = np.zeros((3, self.n))
        matrix[0, 2:] = 1
        matrix[1, 1:-1] = -2
        matrix[2, :-2] = 1
        return matrix / self.dx ** 2

    def vector_d2dx2(self,
                     vector: np.ndarray) -> np.ndarray:
        """Product of input vector and 2nd order derivative operator."""
        return self.diag_tridiag(self.identity(vector), self.d2dx2())

    def x2_d2dx2(self) -> np.ndarray:
        """Product of space coordinate squared and its 2nd order
        derivative operator."""
#        return self.diag_tridiag(
#            self.identity(self.x_grid() ** 2), self.d2dx2())
        return self.vector_d2dx2(self.x_grid() ** 2)

    def propagation(self,
                    dt: float,
                    diff_operator: np.ndarray,
                    v_vector: np.ndarray) -> np.ndarray:
        """Propagation of v_vector for one time step dt."""
        lhs = self.identity() - self.theta * dt * diff_operator
        rhs = self.identity() + (1 - self.theta) * dt * diff_operator
        return solve_banded((1, 1), lhs, self.tridiag_vec(rhs, v_vector))


class Solver:
    """Solve parabolic 1-d PDEs.
    The general structure of the PDEs is
    dV/dt + drift * dV/dx + 1/2 * diffusion^2 * dV^2/dx^2 = rate * V
    The numerical solution is determined using the theta method.
        - theta = 0   : Explicit Euler method
        - theta = 1/2 : Crank-Nicolson method (default)
        - theta = 1   : Fully implicit method
    The grid in the spatial dimension is assumed equidistant.
    """

    def __init__(self, xmin, xmax, nstates, theta=0.5):
        self._xmin = xmin
        self._xmax = xmax
        self._nstates = nstates
        self._theta = theta
        self._dx = (xmax - xmin) / (nstates - 1)
        self._drift_vec = None
        self._diffusion_vec = None
        self._rate_vec = None
        self._identity_mat = None
        self._propagator_mat = None
        self._boundary_vec = None

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def nstates(self):
        return self._nstates

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, val: float):
        self._theta = val

    @property
    def dx(self):
        return self._dx

    def grid(self):
        """Equidistant grid between _xmin and _xmax, including both end
        points.
        """
        return self._dx * np.arange(self._nstates) + self._xmin

    def set_up_drift_vec(self, vector):
        """Drift vector defined by the underlying stochastic process."""
#        self._drift_vec = rate * self.grid()
        self._drift_vec = vector

    def set_up_diffusion_vec(self, vector):
        """Diffusion vector defined by the underlying stochastic
        process.
        """
#        self._diffusion_vec = vol * self.grid()
        self._diffusion_vec = vector

    def set_up_rate_vec(self, vector):
        """Rate vector defined by the underlying stochastic process."""
#        self._rate_vec = rate
        self._rate_vec = vector

    def identity_mat(self):
        """Identity matrix on tri-diagonal form.
            - 1st row: Super-diagonal (not including first element)
            - 2nd row: Diagonal
            - 3rd row: Sub-diagonal (not including last element)
        """
        self._identity_mat = np.zeros((3, self._nstates))
        self._identity_mat[1, :] = 1

    @staticmethod
    def mat_vec_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Product of tri-diagonal matrix and column vector."""
        # Contribution from diagonal
        product = matrix[1, :] * vector
        # Contribution from super-diagonal
        product[:-1] += matrix[0, 1:] * vector[1:]
        # Contribution from sub-diagonal
        product[1:] += matrix[2, :-1] * vector[:-1]
        return product

    def initialization(self):
        """Initialization step..."""
        # Set up identity matrix
        self.identity_mat()

    def set_up_propagator(self):
        """Propagator on tri-diagonal form.
            - 1st row: Super-diagonal (not including first element)
            - 2nd row: Diagonal
            - 3rd row: Sub-diagonal (not including last element)
        """
        diffusion_vec_sq = np.square(self._diffusion_vec)
        dx_sq = self._dx ** 2
        # Super-diagonal, diagonal, and sub-diagonal for interior points
        # Eq. (2.7) - (2.9), L.B.G. Andersen & V.V. Piterbarg 2010
        upper = (self._drift_vec / self._dx + diffusion_vec_sq / dx_sq) / 2
        center = - diffusion_vec_sq / dx_sq - self._rate_vec
        lower = (-self._drift_vec / self._dx + diffusion_vec_sq / dx_sq) / 2
        # Set up propagator matrix consistent with the solve_banded
        # function (scipy.linalg)
        # Eq. (2.11), L.B.G. Andersen & V.V. Piterbarg 2010
        self._propagator_mat = np.zeros((3, self._nstates))
        self._propagator_mat[0, 1:] = upper[:-1]
        self._propagator_mat[1, :] = center
        self._propagator_mat[2, :-1] = lower[1:]

        # Choose Boundary conditions

        # 1) Instrument value is assumed linear in the underlying price
        # process, i.e., dV^2/dx^2 = 0
        # Eq. (2.12) - (2.13), L.B.G. Andersen & V.V. Piterbarg 2010
        self._propagator_mat[1, -1] += 2 * upper[-1]
        self._propagator_mat[2, -2] += - upper[-1]
        self._propagator_mat[1, 0] += 2 * lower[0]
        self._propagator_mat[0, 1] += - lower[0]

        # Set up boundary vector
        self._boundary_vec = np.zeros(self._nstates)

    def propagation(self, dt: float, vector: np.ndarray) -> np.ndarray:
        """Propagation of vector for one time step dt."""
        # Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010
        rhs = self._identity_mat + (1 - self.theta) * dt * self._propagator_mat
        rhs = self.mat_vec_product(rhs, vector) \
            + (1 - self._theta) * self._boundary_vec

        # Update self._propagator_mat and self._boundary_vec
        # UPDATE DIFFUSION_VEC, DRIFT_VEC, and RATE_VEC before method call...
        # should correspond to end of time step...
        self.set_up_propagator()

        # Eq. (2.19), L.B.G. Andersen & V.V. Piterbarg 2010
        rhs += self._theta * self._boundary_vec
        lhs = self._identity_mat - self.theta * dt * self._propagator_mat
        return solve_banded((1, 1), lhs, rhs)

    @staticmethod
    def fd_delta(grid: np.ndarray,
                 function: np.ndarray):
        """Delta determined by central finite difference
        approximation. Assuming equidistant and ascending grid...
        """
        grid_spacing = (grid[1] - grid[0])
        return (function[2:] - function[:-2]) / (2 * grid_spacing)

    @staticmethod
    def fd_gamma(grid: np.ndarray,
                 function: np.ndarray):
        """Gamma determined by finite difference approximation. Assuming
        equidistant and ascending grid...
        """
        grid_spacing = (grid[1] - grid[0])
        return (function[2:] + function[:-2] - 2 * function[1:-1]) \
            / grid_spacing ** 2

    def fd_theta(self,
                 dt: float,
                 function: np.ndarray):
        """Theta determined by central finite difference
        approximation.
        """
        self.set_up_propagator()
        forward = self.propagation(-dt, function)
        backward = self.propagation(dt, function)
        return (forward - backward) / (2 * dt)

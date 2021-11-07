import numpy as np
from scipy.linalg import solve_banded


class Solver:
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

    def x_ddx(self) -> np.ndarray:
        """Product of space coordinate and its 1st order derivative
        operator."""
        return self.diag_tridiag(self.identity(self.x_grid()), self.ddx())

    def d2dx2(self) -> np.ndarray:
        """Finite difference approximation of 2nd order derivative
        operator. At the boundaries, the operator is set equal to
        zero."""
        matrix = np.zeros((3, self.n))
        matrix[0, 2:] = 1
        matrix[1, 1:-1] = -2
        matrix[2, :-2] = 1
        return matrix / self.dx ** 2

    def x2_d2dx2(self) -> np.ndarray:
        """Product of space coordinate squared and its 2nd order
        derivative operator."""
        return self.diag_tridiag(
            self.identity(self.x_grid() ** 2), self.d2dx2())

    def propagation(self,
                    dt: float,
                    diff_operator: np.ndarray,
                    v_vector: np.ndarray) -> np.ndarray:
        """Propagation of v_vector for one time step dt."""
        lhs = self.identity() - self.theta * dt * diff_operator
        rhs = self.identity() + (1 - self.theta) * dt * diff_operator
        return solve_banded((1, 1), lhs, self.tridiag_vec(rhs, v_vector))

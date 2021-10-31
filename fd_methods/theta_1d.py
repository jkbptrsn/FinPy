import numpy as np


class Solver:
    def __init__(self, xmin, xmax, n):
        self._xmin = xmin
        self._xmax = xmax
        self._n = n
        self._dx = (xmin - xmax) / (n - 1)

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
    def dx(self):
        return self._dx

    def x_grid(self):
        return self.dx * np.ndarray(range(n - 1, -1, -1)) + self.xmin

    def identity(self):
        a = np.zeros((self._n, 3))
        a[:, 1] = 1
        return a

    def ddx(self):
        # Finite difference approximation of d/dx
        # Central difference
        a = np.zeros((self._n, 3))
        a[:-2, 0] = -1
        a[2:, 2] = 1
        # Backward difference at boundary
        a[-2, 0] = -2
        a[-1, 1] = 2
        # Forward difference at boundary
        a[1, 2] = 2
        a[0, 1] = -2
        return a / (2 * self.dx)

    def x_ddx(self):
        # Finite difference approximation of x * d/dx
        x = self.x_grid()
        ddx = self.ddx()
        ddx[:-1, 0] *= x[1:]
        ddx[:, 1] *= x
        ddx[1:, 2] *= x[:-1]
        return ddx

    def d2dx2(self):
        # Finite difference approximation of d^2/dx^2
        a = np.zeros((self._n, 3))
        a[:-2, 0] = 1
        a[1:-1, 1] = -2
        a[2:, 2] = 1
        return a / self.dx ** 2

    def x2_d2dx2(self):
        # Finite difference approximation of x^2 * d^2/dx^2
        x = self.x_grid()
        d2dx2 = self.d2dx2()
        d2dx2[:-1, 0] *= x[1:]
        d2dx2[:, 1] *= x
        d2dx2[1:, 2] *= x[:-1]
        return d2dx2

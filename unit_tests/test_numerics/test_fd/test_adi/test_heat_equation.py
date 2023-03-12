import math
import unittest

from matplotlib import pyplot as plt
import numpy as np

from numerics.fd.adi import peaceman_rachford as pr

plot_results = True
print_result = True


def print_to_screen(obj, diff):
    print(obj
          + ": max diff = " + str(diff))


class HeatEquation1D(unittest.TestCase):
    """Numerical solutions of 2-dimensional heat equation.

    The heat equation reads
        dV/dt = k_x * d^2V/dx^2 + k_y * d^2V/dy^2,
    where k_x > 0 is the thermal diffusivity...

    Assuming x in [0, L], t > 0, and defining
        a_n(x) = n * pi * x / L
        b_n(t) = n^2 * pi^2 * k * t / L^2,
    a particular solution is given by
        V_n(t,x) = D_n * sin(a_n(x)) * exp(-b_n(t)),
    where
        D_n = (2 / L) int_0^L f(x) * sin(a_n(x)) dx.
    """

    def setUp(self) -> None:
        self.x_min = 0
        self.x_max = 5
        self.x_steps = 50
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.y_min = 0
        self.y_max = 5
        self.y_steps = 50
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min
        self.band = "tri"
        self.equidistant = True
        self.k = 3
        self.solver = pr.PeachmanRachford2D(self.x_grid, self.y_grid,
                                            self.band, self.equidistant)
        self.zero_matrix = np.zeros((self.x_grid.size, self.y_grid.size))
        self.solver.set_drift(self.zero_matrix)
        self.solver.set_diffusion(math.sqrt(2 * self.k) + self.zero_matrix)
        self.solver.set_rate(self.zero_matrix)
        self.solver.initialization()

    def test_single(self):
        """Single sine function as initial condition.

        The initial condition reads
            f(x) = 6 * sin(a_1(x)),
        and the solution is given by
            V(t,x) = 6 * sin(a_1(x)) * exp(-b_1(t)).
        """
        t_min = 0
        t_max = 0.5
        t_steps = 50
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        ic_y = 6 * np.sin(math.pi * self.y_grid / self.y_max)
        ic = np.outer(ic_x, ic_y)
        for band, limit in (("tri", 5e-3), ("penta", 2e-4)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic.copy()
            # Propagation.
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            analytical_solution = ic \
                * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max) \
                * math.exp(-self.k * (math.pi / self.y_max) ** 2 * t_max)
            if plot_results:
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(self.x_grid, self.y_grid, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Peaceman-Rachford')
                im = ax1.pcolormesh(self.x_grid, self.y_grid, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(3)
            diff = np.abs(analytical_solution - self.solver.solution)
            max_diff = np.max(diff)
            if print_result:
                print_to_screen(str(self), max_diff)
            self.assertTrue(max_diff < limit)

    def test_superposition(self):
        """Superposition of sine functions as initial condition.

        The initial condition reads
            f(x) = 12 * sin(a_3(x)) - 7 * sin(a_2(x)),
        and the solution is given by
            V(t,x) = 12 * sin(a_3(x)) * exp(-b_3(t))
                - 7 * sin(a_2(x)) * exp(-b_2(t)).
        """
        t_min = 0
        t_max = 0.002
        t_steps = 200
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_x1 = 12 * np.sin(9 * math.pi * self.x_grid / self.x_max)
        ic_x2 = - 7 * np.sin(4 * math.pi * self.x_grid / self.x_max)
        ic_x = ic_x1 + ic_x2
        ic_y1 = 12 * np.sin(9 * math.pi * self.y_grid / self.y_max)
        ic_y2 = - 7 * np.sin(4 * math.pi * self.y_grid / self.y_max)
        ic_y = ic_y1 + ic_y2
        ic = np.outer(ic_x, ic_y)
        for band, limit in (("tri", 2e0), ("penta", 1e0)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic.copy()
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            # Analytical solution.
            freq_x1 = self.k * (9 * math.pi / self.x_max) ** 2
            freq_x2 = self.k * (4 * math.pi / self.x_max) ** 2
            solution_x = ic_x1 * math.exp(-freq_x1 * t_max) \
                + ic_x2 * math.exp(-freq_x2 * t_max)
            freq_y1 = self.k * (9 * math.pi / self.y_max) ** 2
            freq_y2 = self.k * (4 * math.pi / self.y_max) ** 2
            solution_y = ic_y1 * math.exp(-freq_y1 * t_max) \
                + ic_y2 * math.exp(-freq_y2 * t_max)
            analytical_solution = np.outer(solution_x, solution_y)
            if plot_results:
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(self.x_grid, self.y_grid, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Peaceman-Rachford')
                im = ax1.pcolormesh(self.x_grid, self.y_grid, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(3)
            diff = np.abs(analytical_solution - self.solver.solution)
            max_diff = np.max(diff)
            if print_result:
                print_to_screen(str(self), max_diff)
            self.assertTrue(max_diff < limit)


if __name__ == '__main__':
    unittest.main()

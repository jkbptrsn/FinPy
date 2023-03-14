import math
import unittest

from matplotlib import pyplot as plt
import numpy as np

from numerics.fd.adi import peaceman_rachford as pr

plot_results = False
print_result = False


def print_to_screen(obj, diff):
    print(obj
          + ": max diff = " + str(diff))


class HeatEquation2D(unittest.TestCase):
    """Numerical solutions of 2-dimensional heat equation.

    The heat equation reads
        dV/dt = k * d^2V/dx^2 + k * d^2V/dy^2,
    where k > 0 is the thermal diffusivity.

    Assuming x in [0, L_x], t > 0, and defining
        a_n(x) = n * pi * x / L_x,
        b_n(t) = n^2 * pi^2 * k * t / L_x^2,
    a particular solution of the x-component is given by
        V_n(t,x) = D_n * sin(a_n(x)) * exp(-b_n(t)),
    where
        D_n = (2 / L_x) int_0^L_x f(x) * sin(a_n(x)) dx.

    The full particular solution reads
        V_{m,n}(t,x,y) = V_{m}(t,x) * V_{n}(t,y)
    """

    def setUp(self) -> None:
        self.x_min = 0
        self.x_max = 4
        self.x_steps = 500
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.y_min = 0
        self.y_max = 7
        self.y_steps = 500
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min
        self.band = "tri"
        self.equidistant = True
        self.k = 3
        self.solver = pr.PeachmanRachford2D(self.x_grid, self.y_grid,
                                            self.band, self.equidistant)
        self.zero_matrix = np.zeros((self.x_grid.size, self.y_grid.size))
        drift_x = self.zero_matrix
        drift_y = self.zero_matrix
        self.solver.set_drift(drift_x, drift_y)
        diffusion_x = math.sqrt(2 * self.k) + self.zero_matrix
        diffusion_y = math.sqrt(2 * self.k) + self.zero_matrix
        self.solver.set_diffusion(diffusion_x, diffusion_y)
        self.solver.set_rate(self.zero_matrix)
        self.solver.initialization()

    def test_single(self):
        """Single sine function as initial condition.

        The initial condition f(x,y) reads
            f(x) = 6 * sin(a_1(x)),
            f(y) = 6 * sin(a_1(y)),
        and the solution V(t,x,y) is the product of
            V(t,x) = 6 * sin(a_1(x)) * exp(-b_1(t)),
            V(t,y) = 6 * sin(a_1(y)) * exp(-b_1(t)).
        """
        t_min = 0
        t_max = 0.5
        t_steps = 15
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        ic_y = 6 * np.sin(math.pi * self.y_grid / self.y_max)
        ic = np.outer(ic_x, ic_y)
        for band, limit in (("tri", 4e-4), ("penta", 4e-4)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic.copy()
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            analytical_solution = ic \
                * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max) \
                * math.exp(-self.k * (math.pi / self.y_max) ** 2 * t_max)
            if plot_results:
                plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(plot_x, plot_y, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Peaceman-Rachford')
                im = ax1.pcolormesh(plot_x, plot_y, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(2)
            # Compare analytical and numerical solutions.
            cutoff = 0.01
            mask = np.abs(analytical_solution) > cutoff
            a_solution = np.where(mask, analytical_solution, cutoff)
            n_solution = np.where(mask, self.solver.solution, cutoff)
            diff = np.abs((a_solution - n_solution) / a_solution)
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
        t_max = 0.005
        t_steps = 30
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
        for band, limit in (("tri", 7e-3), ("penta", 2e-3)):
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
                plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(plot_x, plot_y, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Peaceman-Rachford')
                im = ax1.pcolormesh(plot_x, plot_y, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(2)
            # Compare analytical and numerical solutions.
            cutoff = 1
            mask = np.abs(analytical_solution) > cutoff
            a_solution = np.where(mask, analytical_solution, cutoff)
            n_solution = np.where(mask, self.solver.solution, cutoff)
            diff = np.abs((a_solution - n_solution) / a_solution)
            max_diff = np.max(diff)
            if print_result:
                print_to_screen(str(self), max_diff)
            self.assertTrue(max_diff < limit)


if __name__ == '__main__':
    unittest.main()

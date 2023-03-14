import math
import unittest

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

from numerics.fd.theta import theta
from numerics.fd import misc

plot_function = True
print_result = True


def print_to_screen(obj, diff, relative_diff):
    print(obj
          + ": max diff = " + str(diff)
          + ", relative max diff = " + str(relative_diff))


class HeatEquation1D(unittest.TestCase):
    """Numerical solutions of 1-dimensional heat equation.

    The heat equation reads
        dV/dt = k * d^2V/dx^2,
    where k > 0 is the thermal diffusivity.

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
        self.x_steps = 10000
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.band = "tri"
        self.equidistant = True
        self.theta_parameter = 0.5
        self.k = 3
        self.solver = theta.Theta(self.x_grid, self.band, self.equidistant,
                                  self.theta_parameter)
        self.solver.set_drift(0 * self.x_grid)
        self.solver.set_diffusion(math.sqrt(2 * self.k) + 0 * self.x_grid)
        self.solver.set_rate(0 * self.x_grid)
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
        t_steps = 1000
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        for band, limit in (("tri", 5e-7), ("penta", 2e-8)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic
            if plot_function:
                plt.plot(self.solver.grid, self.solver.solution, "-b")
            # Propagation.
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            analytical_solution = ic \
                * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max)
            if plot_function:
                plt.plot(self.solver.grid, self.solver.solution, "-r")
                plt.pause(2)
                plt.clf()
            diff = np.abs(analytical_solution - self.solver.solution)
            idx_max = np.argmax(diff)
            relative_diff = diff[idx_max] / abs(analytical_solution[idx_max])
            if print_result:
                print_to_screen(str(self), diff[idx_max], relative_diff)
            self.assertTrue(relative_diff < limit)

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
        t_steps = 1000
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_1 = 12 * np.sin(9 * math.pi * self.x_grid / self.x_max)
        ic_2 = - 7 * np.sin(4 * math.pi * self.x_grid / self.x_max)
        for band, limit in (("tri", 2e-7), ("penta", 6e-9)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic_1 + ic_2
            if plot_function:
                plt.plot(self.solver.grid, self.solver.solution, "-b")
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            freq_1 = self.k * (9 * math.pi / self.x_max) ** 2
            freq_2 = self.k * (4 * math.pi / self.x_max) ** 2
            analytical_solution = ic_1 * math.exp(-freq_1 * t_max) \
                + ic_2 * math.exp(-freq_2 * t_max)
            if plot_function:
                plt.plot(self.solver.grid, self.solver.solution, "-r")
                plt.pause(2)
                plt.clf()
            diff = np.abs(analytical_solution - self.solver.solution)
            idx_max = np.argmax(diff)
            relative_diff = diff[idx_max] / abs(analytical_solution[idx_max])
            if print_result:
                print_to_screen(str(self), diff[idx_max], relative_diff)
            self.assertTrue(relative_diff < limit)

    def test_single_convergence(self):
        """..."""
        t_min = 0
        t_max = 0.5

        t_steps_min = 100
        t_steps_max = 1000
        t_steps_int = 200
        n_points = len(range(t_steps_min, t_steps_max, t_steps_int))

        # Initial condition.
        ic = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        # Analytical solution.
        analytical_solution = ic * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max)

        dt_array = np.zeros(n_points)
        norms_array = np.zeros((3, n_points))

        count = 0
        for t_steps in range(t_steps_min, t_steps_max, t_steps_int):
            dt = (t_max - t_min) / (t_steps - 1)
            t_grid = dt * np.arange(t_steps) + t_min
            self.solver.solution = ic
            # Propagation.
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            norms = misc.norms_1d(analytical_solution, self.solver.solution, self.dx, slice_nr=1)
            dt_array[count] = dt
            norms_array[:, count] = norms
            count += 1

#        print(dt_array)
#        print(norms_array)

        if plot_function:
            plt.plot(np.log(dt_array), np.log(norms_array[0, :]), "ob")
            plt.plot(np.log(dt_array), np.log(norms_array[1, :]), "or")
            plt.plot(np.log(dt_array), np.log(norms_array[2, :]), "ok")
            plt.pause(2)
            plt.clf()

        lr1 = linregress(np.log(dt_array), np.log(norms_array[0, :]))
        lr2 = linregress(np.log(dt_array), np.log(norms_array[1, :]))
        lr3 = linregress(np.log(dt_array), np.log(norms_array[2, :]))
        if print_result:
            print(lr1.slope, lr2.slope, lr3.slope)

    def test_superposition_convergence(self):
        pass


if __name__ == '__main__':
    unittest.main()

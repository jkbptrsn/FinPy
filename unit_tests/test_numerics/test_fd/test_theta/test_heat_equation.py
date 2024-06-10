import math
import unittest

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

from numerics.fd.theta import theta
from numerics.fd import misc

plot_result = False
print_result = False


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
        self.x_steps = 1000
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
        t_steps = 500
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        for band, limit in (("tri", 5e-7), ("penta", 7e-8)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic
            if plot_result:
                plt.plot(self.solver.grid, self.solver.solution, "-b")
            # Propagation.
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            analytical_solution = ic \
                * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max)
            if plot_result:
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
        t_steps = 500
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_1 = 12 * np.sin(9 * math.pi * self.x_grid / self.x_max)
        ic_2 = - 7 * np.sin(4 * math.pi * self.x_grid / self.x_max)
        for band, limit in (("tri", 2e-5), ("penta", 2e-8)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic_1 + ic_2
            if plot_result:
                plt.plot(self.solver.grid, self.solver.solution, "-b")
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            freq_1 = self.k * (9 * math.pi / self.x_max) ** 2
            freq_2 = self.k * (4 * math.pi / self.x_max) ** 2
            analytical_solution = ic_1 * math.exp(-freq_1 * t_max) \
                + ic_2 * math.exp(-freq_2 * t_max)
            if plot_result:
                plt.plot(self.solver.grid, self.solver.solution, "-r")
                plt.pause(2)
                plt.clf()
            diff = np.abs(analytical_solution - self.solver.solution)
            idx_max = np.argmax(diff)
            relative_diff = diff[idx_max] / abs(analytical_solution[idx_max])
            if print_result:
                print_to_screen(str(self), diff[idx_max], relative_diff)
            self.assertTrue(relative_diff < limit)

    def test_convergence_in_space(self):
        """Test convergence rate in spatial dimension."""
        for test in ("single", "superposition"):
            t_min = 0
            if test == "single":
                t_max = 0.5
            else:
                t_max = 0.005
            t_steps = 100
            dt = (t_max - t_min) / (t_steps - 1)
            t_grid = dt * np.arange(t_steps) + t_min
            # Number of states.
            x_states_start = 201
            x_states_array = np.arange(x_states_start, 2 * x_states_start, 50)
            # Number of times the number of grid points is doubled.
            n_doubling = 3
            # Arrays for storing data.
            step_array = np.zeros((n_doubling - 1) * x_states_array.size)
            norm_array = np.zeros((3, (n_doubling - 1) * x_states_array.size))
            counter = 0
            for m in x_states_array:
                x_states = m
                for n in range(n_doubling):
                    # Set up PDE solver.
                    self.x_steps = x_states
                    self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
                    self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
                    self.band = "tri"
                    self.equidistant = True
                    self.theta_parameter = 0.5
                    self.k = 3
                    self.solver = \
                        theta.Theta(self.x_grid, self.band,
                                    self.equidistant, self.theta_parameter)
                    self.solver.set_drift(0 * self.x_grid)
                    diffusion = math.sqrt(2 * self.k) + 0 * self.x_grid
                    self.solver.set_diffusion(diffusion)
                    self.solver.set_rate(0 * self.x_grid)
                    self.solver.initialization()

                    # Initial condition.
                    if test == "single":
                        ic = 6 * np.sin(math.pi * self.x_grid / self.x_max)
                        self.solver.solution = ic
                    else:
                        ic_1 = 12 * np.sin(9 * math.pi * self.x_grid / self.x_max)
                        ic_2 = - 7 * np.sin(4 * math.pi * self.x_grid / self.x_max)
                        self.solver.solution = ic_1 + ic_2

                    # Propagation.
                    for time_step in np.diff(t_grid):
                        self.solver.propagation(time_step)

                    # Save result.
                    solution = self.solver.solution

                    # Calculate norms.
                    if n != 0:
                        step_array[counter] = np.log(self.dx)
                        norm_array[:, counter] = \
                            misc.norms_1d(solution_old, solution, dx_old)
                        norm_array[:, counter] = np.log(norm_array[:, counter])
                        counter += 1
                    # Save result.
                    solution_old = solution
                    dx_old = self.dx
                    # Update grid spacing in spatial dimension.
                    x_states = 2 * x_states - 1
            if plot_result:
                fig, ax = plt.subplots(nrows=3, ncols=1)
                ax[0].plot(step_array, norm_array[0, :],
                           "ok", label="Center norm")
                title = "Convergence tests wrt step size in space dimension"
                ax[0].set_title(title)
                ax[0].legend()
                ax[0].label_outer()
                ax[1].plot(step_array, norm_array[1, :],
                           "or", label="Max norm")
                ax[1].set(ylabel="log(norm)")
                ax[1].legend()
                ax[1].label_outer()
                ax[2].plot(step_array, norm_array[2, :],
                           "ob", label="L2 norm")
                ax[2].set(xlabel="log(Delta x)")
                ax[2].legend()
                plt.show(block=False)
                plt.pause(2)
                plt.clf()
            # Linear regression
            lr1 = linregress(step_array, norm_array[0, :])
            lr2 = linregress(step_array, norm_array[1, :])
            lr3 = linregress(step_array, norm_array[2, :])
            if print_result:
                print(self, lr1.slope, lr2.slope, lr3.slope)
            self.assertTrue(abs(lr1.slope - 2) < 6e-4)
            self.assertTrue(abs(lr2.slope - 2) < 6e-4)
            self.assertTrue(abs(lr3.slope - 2) < 6e-4)

    def test_convergence_in_time(self):
        """Test convergence rate in time dimension."""
        for test in ("single", "superposition"):
            t_min = 0
            if test == "single":
                t_max = 0.5
            else:
                t_max = 0.005

            self.x_steps = 100
            self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
            self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
            self.band = "tri"
            self.equidistant = True
            self.theta_parameter = 0.5
            self.k = 3
            self.solver = \
                theta.Theta(self.x_grid, self.band,
                            self.equidistant, self.theta_parameter)
            self.solver.set_drift(0 * self.x_grid)
            diffusion = math.sqrt(2 * self.k) + 0 * self.x_grid
            self.solver.set_diffusion(diffusion)
            self.solver.set_rate(0 * self.x_grid)
            self.solver.initialization()

            # Number of states.
            t_steps_array = np.array([101, 151])
            # Number of times the number of grid points is doubled.
            n_doubling = 4
            # Arrays for storing data.
            step_array = np.zeros((n_doubling - 1) * t_steps_array.size)
            norm_array = np.zeros((3, (n_doubling - 1) * t_steps_array.size))
            counter = 0
            for m in t_steps_array:
                t_steps = m
                dt = (t_max - t_min) / (t_steps - 1)
                for n in range(n_doubling):

                    t_grid = dt * np.arange(t_steps) + t_min

                    # Initial condition.
                    if test == "single":
                        ic = 6 * np.sin(math.pi * self.x_grid / self.x_max)
                        self.solver.solution = ic
                    else:
                        ic_1 = 12 * np.sin(9 * math.pi * self.x_grid / self.x_max)
                        ic_2 = - 7 * np.sin(4 * math.pi * self.x_grid / self.x_max)
                        self.solver.solution = ic_1 + ic_2

                    # Propagation.
                    for time_step in np.diff(t_grid):
                        self.solver.propagation(time_step)

                    # Save result.
                    solution = self.solver.solution

                    # Calculate norms.
                    if n != 0:
                        step_array[counter] = np.log(dt)
                        norm_array[:, counter] = \
                            misc.norms_1d(solution_old, solution,
                                          dx_old, slice_nr=1)
                        norm_array[:, counter] = np.log(norm_array[:, counter])
                        counter += 1
                    # Save result.
                    solution_old = solution
                    dx_old = self.dx

                    # Update grid spacing in time dimension.
                    t_steps = (t_steps - 1) * 2 + 1
                    dt = (t_max - t_min) / (t_steps - 1)
            if plot_result:
                fig, ax = plt.subplots(nrows=3, ncols=1)
                ax[0].plot(step_array, norm_array[0, :],
                           "ok", label="Center norm")
                title = "Convergence tests wrt step size in time dimension"
                ax[0].set_title(title)
                ax[0].legend()
                ax[0].label_outer()
                ax[1].plot(step_array, norm_array[1, :],
                           "or", label="Max norm")
                ax[1].set(ylabel="log(norm)")
                ax[1].legend()
                ax[1].label_outer()
                ax[2].plot(step_array, norm_array[2, :],
                           "ob", label="L2 norm")
                ax[2].set(xlabel="log(Delta t)")
                ax[2].legend()
                plt.show(block=False)
                plt.pause(2)
                plt.clf()
            # Linear regression
            lr1 = linregress(step_array, norm_array[0, :])
            lr2 = linregress(step_array, norm_array[1, :])
            lr3 = linregress(step_array, norm_array[2, :])
            if print_result:
                print(self, lr1.slope, lr2.slope, lr3.slope)
            self.assertTrue(abs(lr1.slope - 2) < 1e-5)
            self.assertTrue(abs(lr2.slope - 2) < 1e-5)
            self.assertTrue(abs(lr3.slope - 2) < 1e-5)

import math
import unittest

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

from numerics.fd.adi import craig_sneyd as cs
from numerics.fd.adi import douglas_rachford as dr
from numerics.fd.adi import peaceman_rachford as pr
from numerics.fd import misc

plot_result = False
print_result = False


def print_to_screen(obj, diff):
    print(obj
          + ": max diff = " + str(diff))


class HeatEquation2DPeacemanRachford(unittest.TestCase):
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
        self.x_steps = 200
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.y_min = 0
        self.y_max = 5
        self.y_steps = 200
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min
        self.band = "tri"
        self.equidistant = True
        self.k = 3

        self.solver = pr.PeachmanRachford2D(self.x_grid, self.y_grid,
                                            self.band, self.equidistant)
#        self.solver = dr.DouglasRachford2D(self.x_grid, self.y_grid,
#                                           self.band, self.equidistant)
#        self.solver = cs.CraigSneyd2D(self.x_grid, self.y_grid,
#                                      self.band, self.equidistant)

        self.zero_matrix = np.zeros((self.x_grid.size, self.y_grid.size))
        drift_x = self.zero_matrix
        drift_y = self.zero_matrix
        self.solver.set_drift(drift_x, drift_y)
        diffusion_x = math.sqrt(2 * self.k) + self.zero_matrix
        diffusion_y = math.sqrt(2 * self.k) + self.zero_matrix
        self.solver.set_diffusion(diffusion_x, diffusion_y)
        self.solver.set_rate(self.zero_matrix)
        self.solver.initialization()

    def test_superposition(self):
        """Superposition of sine functions as initial condition.

        The initial condition f(x,y) reads
            f(x) = 6 * sin(a_1(x)),
            f(y) = 12 * sin(a_3(y)) - 7 * sin(a_2(y)),
        and the solution V(t,x,y) is the product of
            V(t,x) = 6 * sin(a_1(x)) * exp(-b_1(t)),
            V(t,y) = 12 * sin(a_3(y)) * exp(-b_3(t))
                - 7 * sin(a_2(y)) * exp(-b_2(t)).
        """
        t_min = 0
        t_max = 0.005
        t_steps = 50
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        ic_y1 = 12 * np.sin(9 * math.pi * self.y_grid / self.y_max)
        ic_y2 = - 7 * np.sin(4 * math.pi * self.y_grid / self.y_max)
        ic_y = ic_y1 + ic_y2
        ic = np.outer(ic_x, ic_y)
        for band, limit in (("tri", 7e-3), ("penta", 3e-5)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic.copy()
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            solution_x = \
                ic_x * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max)
            freq_y1 = self.k * (9 * math.pi / self.y_max) ** 2
            freq_y2 = self.k * (4 * math.pi / self.y_max) ** 2
            solution_y = ic_y1 * math.exp(-freq_y1 * t_max) \
                + ic_y2 * math.exp(-freq_y2 * t_max)
            analytical_solution = np.outer(solution_x, solution_y)
            if plot_result:
                plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(plot_x, plot_y, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Peaceman-Rachford')
                im = ax1.pcolormesh(plot_x, plot_y, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(2)
                plt.clf()
            # Compare analytical and numerical solutions.
            cutoff = 5
            mask = np.abs(analytical_solution) > cutoff
            a_solution = np.where(mask, analytical_solution, cutoff)
            n_solution = np.where(mask, self.solver.solution, cutoff)
            diff = np.abs((a_solution - n_solution) / a_solution)
            max_diff = np.max(diff)
            if print_result:
                print_to_screen(str(self), max_diff)
            self.assertTrue(max_diff < limit)

    def test_convergence_in_space(self):
        """Test convergence rate in spatial dimension."""
        t_min = 0
        t_max = 0.005
        t_steps = 50
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Number of states.
        x_states_array = np.array([101, 151])
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
#                self.y_steps = x_states
#                self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
#                self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min

                self.solver = pr.PeachmanRachford2D(self.x_grid, self.y_grid,
                                                    self.band, self.equidistant)
#                self.solver = dr.DouglasRachford2D(self.x_grid, self.y_grid,
#                                                   self.band, self.equidistant)
#                self.solver = cs.CraigSneyd2D(self.x_grid, self.y_grid,
#                                              self.band, self.equidistant)

                self.zero_matrix = np.zeros((self.x_grid.size, self.y_grid.size))
                drift_x = self.zero_matrix
                drift_y = self.zero_matrix
                self.solver.set_drift(drift_x, drift_y)
                diffusion_x = math.sqrt(2 * self.k) + self.zero_matrix
                diffusion_y = math.sqrt(2 * self.k) + self.zero_matrix
                self.solver.set_diffusion(diffusion_x, diffusion_y)
                self.solver.set_rate(self.zero_matrix)
                self.solver.initialization()

                # Initial condition.
                ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
                ic_y1 = 12 * np.sin(9 * math.pi * self.y_grid / self.y_max)
                ic_y2 = - 7 * np.sin(4 * math.pi * self.y_grid / self.y_max)
                ic_y = ic_y1 + ic_y2
                ic = np.outer(ic_x, ic_y)
                self.solver.solution = ic
                # Propagation.
                for time_step in np.diff(t_grid):
                    self.solver.propagation(time_step)
                # Save result.
                solution = self.solver.solution
                # Calculate norms.
                if n != 0:

                    step_array[counter] = np.log(self.dx)
                    norm_array[:, counter] = \
                        misc.norms_2d(solution_old, solution, (dx_old, dy_old),
                                      direction="x")
#                    step_array[counter] = np.log(self.dy)
#                    norm_array[:, counter] = \
#                        misc.norms_2d(solution_old, solution, (dx_old, dy_old),
#                                      direction="y")

                    norm_array[:, counter] = np.log(norm_array[:, counter])
                    counter += 1
                # Save result.
                solution_old = solution
                dx_old = self.dx
                dy_old = self.dy
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
        self.assertTrue(abs(lr1.slope - 2) < 2e-3)
        self.assertTrue(abs(lr2.slope - 2) < 2e-3)
        self.assertTrue(abs(lr3.slope - 2) < 2e-3)

    def test_convergence_in_time(self):
        """Test convergence rate in time dimension."""
        t_min = 0
        t_max = 0.005
        # Number of states.
        t_steps_array = np.array([51, 101, 151, 201])
        # Number of times the number of grid points is doubled.
        n_doubling = 2
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
                ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
                ic_y1 = 12 * np.sin(9 * math.pi * self.y_grid / self.y_max)
                ic_y2 = - 7 * np.sin(4 * math.pi * self.y_grid / self.y_max)
                ic_y = ic_y1 + ic_y2
                ic = np.outer(ic_x, ic_y)
                self.solver.solution = ic
                # Propagation.
                for time_step in np.diff(t_grid):
                    self.solver.propagation(time_step)
                # Save result.
                solution = self.solver.solution

                # Calculate norms.
                if n != 0:

                    step_array[counter] = np.log(dt)
                    norm_array[:, counter] = \
                        misc.norms_2d(solution_old, solution, (dx_old, dy_old),
                                      direction="x", slice_nr=1)

                    norm_array[:, counter] = np.log(norm_array[:, counter])
                    counter += 1
                # Save result.
                solution_old = solution
                dx_old = self.dx
                dy_old = self.dy

                # Update grid spacing in time dimension.
                t_steps = (t_steps - 1) * 2 + 1
                dt = (t_max - t_min) / (t_steps - 1)

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
        self.assertTrue(abs(lr1.slope - 2) < 2e3)
        self.assertTrue(abs(lr2.slope - 2) < 2e3)
        self.assertTrue(abs(lr3.slope - 2) < 2e3)


class HeatEquation2DDouglasRachford(unittest.TestCase):
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
        self.x_steps = 200
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.y_min = 0
        self.y_max = 5
        self.y_steps = 200
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min
        self.band = "tri"
        self.equidistant = True
        self.k = 3
        self.solver = dr.DouglasRachford2D(self.x_grid, self.y_grid,
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

    def test_superposition(self):
        """Superposition of sine functions as initial condition.

        The initial condition f(x,y) reads
            f(x) = 6 * sin(a_1(x)),
            f(y) = 12 * sin(a_3(y)) - 7 * sin(a_2(y)),
        and the solution V(t,x,y) is the product of
            V(t,x) = 6 * sin(a_1(x)) * exp(-b_1(t)),
            V(t,y) = 12 * sin(a_3(y)) * exp(-b_3(t))
                - 7 * sin(a_2(y)) * exp(-b_2(t)).
        """
        t_min = 0
        t_max = 0.005
        t_steps = 50
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        ic_y1 = 12 * np.sin(9 * math.pi * self.y_grid / self.y_max)
        ic_y2 = - 7 * np.sin(4 * math.pi * self.y_grid / self.y_max)
        ic_y = ic_y1 + ic_y2
        ic = np.outer(ic_x, ic_y)
        for band, limit in (("tri", 7e-3), ("penta", 3e-5)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic.copy()
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            solution_x = \
                ic_x * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max)
            freq_y1 = self.k * (9 * math.pi / self.y_max) ** 2
            freq_y2 = self.k * (4 * math.pi / self.y_max) ** 2
            solution_y = ic_y1 * math.exp(-freq_y1 * t_max) \
                + ic_y2 * math.exp(-freq_y2 * t_max)
            analytical_solution = np.outer(solution_x, solution_y)
            if plot_result:
                plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(plot_x, plot_y, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Douglas-Rachford')
                im = ax1.pcolormesh(plot_x, plot_y, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(2)
            # Compare analytical and numerical solutions.
            cutoff = 5
            mask = np.abs(analytical_solution) > cutoff
            a_solution = np.where(mask, analytical_solution, cutoff)
            n_solution = np.where(mask, self.solver.solution, cutoff)
            diff = np.abs((a_solution - n_solution) / a_solution)
            max_diff = np.max(diff)
            if print_result:
                print_to_screen(str(self), max_diff)
            self.assertTrue(max_diff < limit)

    def test_convergence(self):
        pass


class HeatEquation2DCraigSneyd(unittest.TestCase):
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
        self.x_steps = 200
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.y_min = 0
        self.y_max = 5
        self.y_steps = 200
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min
        self.band = "tri"
        self.equidistant = True
        self.k = 3
        self.solver = cs.CraigSneyd2D(self.x_grid, self.y_grid,
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

    def test_superposition(self):
        """Superposition of sine functions as initial condition.

        The initial condition f(x,y) reads
            f(x) = 6 * sin(a_1(x)),
            f(y) = 12 * sin(a_3(y)) - 7 * sin(a_2(y)),
        and the solution V(t,x,y) is the product of
            V(t,x) = 6 * sin(a_1(x)) * exp(-b_1(t)),
            V(t,y) = 12 * sin(a_3(y)) * exp(-b_3(t))
                - 7 * sin(a_2(y)) * exp(-b_2(t)).
        """
        t_min = 0
        t_max = 0.005
        t_steps = 50
        dt = (t_max - t_min) / (t_steps - 1)
        t_grid = dt * np.arange(t_steps) + t_min
        # Initial condition.
        ic_x = 6 * np.sin(math.pi * self.x_grid / self.x_max)
        ic_y1 = 12 * np.sin(9 * math.pi * self.y_grid / self.y_max)
        ic_y2 = - 7 * np.sin(4 * math.pi * self.y_grid / self.y_max)
        ic_y = ic_y1 + ic_y2
        ic = np.outer(ic_x, ic_y)
        for band, limit in (("tri", 7e-3), ("penta", 3e-5)):
            self.solver.band = band
            self.solver.initialization()
            self.solver.solution = ic.copy()
            for time_step in np.diff(t_grid):
                self.solver.propagation(time_step)
            solution_x = \
                ic_x * math.exp(-self.k * (math.pi / self.x_max) ** 2 * t_max)
            freq_y1 = self.k * (9 * math.pi / self.y_max) ** 2
            freq_y2 = self.k * (4 * math.pi / self.y_max) ** 2
            solution_y = ic_y1 * math.exp(-freq_y1 * t_max) \
                + ic_y2 * math.exp(-freq_y2 * t_max)
            analytical_solution = np.outer(solution_x, solution_y)
            if plot_result:
                plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(plot_x, plot_y, self.solver.solution)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Craig-Sneyd')
                im = ax1.pcolormesh(plot_x, plot_y, analytical_solution)
                fig.colorbar(im, ax=ax1)
                ax1.set_title('Analytical')
                plt.pause(2)
            # Compare analytical and numerical solutions.
            cutoff = 5
            mask = np.abs(analytical_solution) > cutoff
            a_solution = np.where(mask, analytical_solution, cutoff)
            n_solution = np.where(mask, self.solver.solution, cutoff)
            diff = np.abs((a_solution - n_solution) / a_solution)
            max_diff = np.max(diff)
            if print_result:
                print_to_screen(str(self), max_diff)
            self.assertTrue(max_diff < limit)

    def test_convergence(self):
        pass


if __name__ == '__main__':
    unittest.main()

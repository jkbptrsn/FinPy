from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
import unittest

import utils.payoffs as payoffs
import numerical_methods.finite_difference.theta as theta

plot_results = False
print_results = True

# Parameters.
rate = 0.1
strike = 0.5
vol = 0.05
expiry = 10
kappa = 2
mean_rate = 0.05
solver_type = "Andersen"


class Theta1D(unittest.TestCase):

    def test_zcbond_in_space(self):
        """Test fully implicit method and Crank-Nicolson method."""
        # Choose theta method.
        for theta_factor in (1, 0.5):
            # Time dimension.
            t_min = 0
            t_max = expiry
            t_steps = 1001
            dt = (t_max - t_min) / (t_steps - 1)
            # Spatial dimension.
            x_min = -0.5
            x_max = 0.5
            # Number of states.
            x_states_start = 501
            x_states_array = np.arange(x_states_start, 2 * x_states_start, 100)
            # Number of times the number of grid points is doubled.
            n_doubling = 4
            # Arrays for storing data.
            step_array = np.zeros((n_doubling - 1) * x_states_array.size)
            norm_array = np.zeros((3, (n_doubling - 1) * x_states_array.size))
            counter = 0
            for m in x_states_array:
                x_states = m
                for n in range(n_doubling):
                    # Set up PDE solver.
                    # TODO: The test fails for Andreasen implementation...
                    solver = theta.setup_vasicek(x_min, x_max, x_states,
                                                 dt, kappa, mean_rate, vol,
                                                 theta=theta_factor,
                                                 method=solver_type)
                    # Terminal solution.
                    solver.solution = payoffs.zero_coupon_bond(solver.grid())
                    # Initialization.
                    solver.initialization()
                    # Backward propagation to time zero.
                    for t in range(t_steps - 1):
                        solver.propagation()
                    # Save result.
                    solution = solver.solution
                    # Calculate norms.
                    if n != 0:
                        step_array[counter] = np.log(solver.dx)
                        norm_array[:, counter] = \
                            theta.norm_diff_1d(solution_old, solution, dx_old)
                        norm_array[:, counter] = np.log(norm_array[:, counter])
                        counter += 1
                    # Save result.
                    solution_old = solution
                    dx_old = solver.dx
                    # Update grid spacing in spatial dimension.
                    x_states = 2 * x_states - 1
            if plot_results:
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
                plt.show()
                plt.pause(5)
            # Linear regression
            lr1 = linregress(step_array, norm_array[0, :])
            lr2 = linregress(step_array, norm_array[1, :])
            lr3 = linregress(step_array, norm_array[2, :])
            if print_results:
                print(lr1.slope, lr2.slope, lr3.slope)
            self.assertTrue(abs(lr1.slope - 2) < 3e-3)
            self.assertTrue(abs(lr2.slope - 2) < 1e-3)
            self.assertTrue(abs(lr3.slope - 2) < 4e-2)

    def test_zcbond_in_time(self):
        """Test fully implicit method and Crank-Nicolson method. Works
        correctly for both Andersen and Andreasen implementations...
        """
        # Choose theta method.
        for theta_factor in (1, 0.5):
            # Time dimension.
            t_min = 0
            t_max = expiry
            # Spatial dimension.
            x_min = -0.5
            x_max = 0.5
            x_states = 201
            # Number of states.
            t_steps_array = np.array([501, 601, 701, 801, 901])
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
                    # Set up PDE solver.
                    solver = theta.setup_vasicek(x_min, x_max, x_states,
                                                 dt, kappa, mean_rate, vol,
                                                 theta=theta_factor,
                                                 method=solver_type)
                    # Terminal solution.
                    solver.solution = payoffs.zero_coupon_bond(solver.grid())
                    # Initialization.
                    solver.initialization()
                    # Backward propagation to time zero.
                    for t in range(t_steps - 1):
                        solver.propagation()
                    # Save result.
                    solution = solver.solution
                    # Calculate norms.
                    if n != 0:
                        step_array[counter] = np.log(dt)
                        norm_array[:, counter] = \
                            theta.norm_diff_1d(solution_old, solution,
                                               dx_old, slice_nr=1)
                        norm_array[:, counter] = np.log(norm_array[:, counter])
                        counter += 1
                    # Save result.
                    solution_old = solution
                    dx_old = solver.dx
                    # Update grid spacing in time dimension.
                    t_steps = (t_steps - 1) * 2 + 1
                    dt = (t_max - t_min) / (t_steps - 1)
            if plot_results:
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
                plt.show()
                plt.pause(5)
            # Linear regression
            lr1 = linregress(step_array, norm_array[0, :])
            lr2 = linregress(step_array, norm_array[1, :])
            lr3 = linregress(step_array, norm_array[2, :])
            if print_results:
                print(lr1.slope, lr2.slope, lr3.slope)
            order = 2
            if theta_factor == 1:
                order = 1
            self.assertTrue(abs(lr1.slope - order) < 1e-3)
            self.assertTrue(abs(lr2.slope - order) < 1e-3)
            self.assertTrue(abs(lr3.slope - order) < 1e-3)


if __name__ == '__main__':
    unittest.main()

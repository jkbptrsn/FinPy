import unittest

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

from models.black_scholes import call_option as call
from numerics.fd import misc

plot_results = False
print_results = False

# Parameters.
rate = 0.04
strike = 100
vol = 0.3
expiry = 1


class Theta1D(unittest.TestCase):

    def test_call_in_space(self):
        """Test fully implicit method and Crank-Nicolson method."""
        # Choose theta method.
        for theta_factor in (1, 0.5):
            # Time dimension.
            t_min = 0
            t_max = expiry
            t_steps = 301
            dt = (t_max - t_min) / (t_steps - 1)
            # Spatial dimension.
            # TODO: Why should the interval be symmetric about the strike?
            x_min = 10
            x_max = 190
            # Number of states.
            x_states_start = 51
            x_states_array = np.arange(x_states_start, 2 * x_states_start, 10)
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
                    event_grid = dt * np.arange(t_steps) - t_min
                    expiry_idx = t_steps - 1
                    instrument = \
                        call.Call(rate, vol, strike, expiry_idx, event_grid)
                    dx = (x_max - x_min) / (x_states - 1)
                    x_grid = dx * np.arange(x_states) + x_min
                    instrument.fd_setup(x_grid, equidistant=True,
                                        theta_value=theta_factor)
                    # Backward propagation to time zero.
                    instrument.fd_solve()
                    # Save result.
                    solution = instrument.fd.solution
                    # Calculate norms.
                    if n != 0:
                        step_array[counter] = np.log(dx)
                        norm_array[:, counter] = \
                            misc.norms_1d(solution_old, solution, dx_old)
                        norm_array[:, counter] = np.log(norm_array[:, counter])
                        counter += 1
                    # Save result.
                    solution_old = solution
                    dx_old = dx
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
                plt.show(block=False)
                plt.pause(2)
            # Linear regression
            lr1 = linregress(step_array, norm_array[0, :])
            lr2 = linregress(step_array, norm_array[1, :])
            lr3 = linregress(step_array, norm_array[2, :])
            if print_results:
                print(lr1.slope, lr2.slope, lr3.slope)
            self.assertTrue(abs(lr1.slope - 2) < 2e-3)
            self.assertTrue(abs(lr2.slope - 2) < 2e-3)
            self.assertTrue(abs(lr3.slope - 2) < 2e-3)

    def test_call_in_time(self):
        """Test fully implicit method and Crank-Nicolson method."""
        # Choose theta method.
        for theta_factor in (1, 0.5):
            # Time dimension.
            t_min = 0
            t_max = expiry
            # Spatial dimension.
            # TODO: Why should the interval be asymmetric about the strike?
            x_min = 9
            x_max = 190
            x_states = 101
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
                    # Set up PDE solver.
                    event_grid = dt * np.arange(t_steps) + t_min
                    expiry_idx = t_steps - 1
                    instrument = \
                        call.Call(rate, vol, strike, expiry_idx, event_grid)
                    dx = (x_max - x_min) / (x_states - 1)
                    x_grid = dx * np.arange(x_states) + x_min
                    instrument.fd_setup(x_grid, equidistant=True,
                                        theta_value=theta_factor)
                    # Backward propagation to time zero.
                    instrument.fd_solve()
                    # Save result.
                    solution = instrument.fd.solution
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
                    dx_old = dx
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
                plt.show(block=False)
                plt.pause(2)
            # Linear regression
            lr1 = linregress(step_array, norm_array[0, :])
            lr2 = linregress(step_array, norm_array[1, :])
            lr3 = linregress(step_array, norm_array[2, :])
            if print_results:
                print(lr1.slope, lr2.slope, lr3.slope)
            order = 2
            if theta_factor == 1:
                order = 1
            self.assertTrue(abs(lr1.slope - order) < 2e-3)
            self.assertTrue(abs(lr2.slope - order) < 2e-3)
            self.assertTrue(abs(lr3.slope - order) < 2e-3)


if __name__ == '__main__':
    unittest.main()

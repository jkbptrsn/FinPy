import math
import unittest

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from models.heston import call_option as call

plot_result = False
print_result = False


class CallOption(unittest.TestCase):
    """European call option in Heston model."""

    def setUp(self) -> None:
        # Model parameters.
        self.rate = 0.03
        self.kappa = 3
        self.eta = 0.12
        self.vol = 0.041
        self.correlation = 0.6
        self.strike = 100
        self.band = "tri"
        self.equidistant = False
        # Event grid.
        self.t_min = 0
        self.t_max = 1
        self.t_steps = 31
        self.dt = (self.t_max - self.t_min) / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps) + self.t_min
        # Option expiry index.
        self.expiry_idx = self.event_grid.size - 1
        # Call option object.
        self.instrument = call.Call(
            self.rate, self.kappa, self.eta, self.vol, self.correlation,
            self.strike, self.expiry_idx, self.event_grid)
        # Call option object with zero correlation.
        self.instrument_zero = call.Call(
            self.rate, self.kappa, self.eta, self.vol, 0,
            self.strike, self.expiry_idx, self.event_grid)
        # FD spatial grids.
        self.x_min = 2
        self.x_max = 400
        self.x_steps = 51
        self.y_min = 0
        self.y_max = 1
        self.y_steps = 21
        if self.equidistant:
            self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
            self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
            self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
            self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min
        else:
            const_c = 10
            d_eps = (math.asinh((self.x_max - self.strike) / const_c)
                     - math.asinh(-self.strike / const_c)) / self.x_steps
            eps_grid = d_eps * np.arange(self.x_steps) \
                + math.asinh(-self.strike / const_c)
            x_grid = self.strike + const_c * np.sinh(eps_grid)
            self.x_grid = x_grid[1:]
            const_d = self.y_max / 500
            d_eta = math.asinh(self.y_max / const_d) / (self.y_steps - 1)
            eta_grid = d_eta * np.arange(self.y_steps)
            y_grid = const_d * np.sinh(eta_grid)
            self.y_grid = y_grid
        # Set up FD solvers.
        self.instrument.fd_setup(self.x_grid, self.y_grid)
        self.instrument_zero.fd_setup(self.x_grid, self.y_grid)

    def test_craig_sneyd(self):
        """Finite difference pricing of European call option."""
        # Numerical result.
        self.instrument.fd_solve()
        self.instrument_zero.fd_solve()
        # Analytical result.
        a_result = np.zeros(self.instrument.fd.solution.shape)
        a_result_zero = np.zeros(self.instrument_zero.fd.solution.shape)
        for idx_x, x in enumerate(self.instrument.fd.grid_x):
            for idx_y, y in enumerate(self.instrument.fd.grid_y):
                a_result[idx_x, idx_y] = (
                    self.instrument.price(x, y, 0))
                a_result_zero[idx_x, idx_y] = (
                    self.instrument_zero.price(x, y, 0))
        diff_n = self.instrument.fd.solution - self.instrument_zero.fd.solution
        diff_a = a_result - a_result_zero
        max_diff = np.max(np.abs(diff_n - diff_a))
        if print_result:
            print(max_diff, 0.09 * np.max(np.abs(diff_a)))
        self.assertTrue(max_diff < 0.09 * np.max(np.abs(diff_a)))
        if plot_result:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            diff = (self.instrument.fd.solution
                    - self.instrument_zero.fd.solution)
            ax.plot_surface(plot_x, plot_y, diff, cmap=cm.jet)
            ax.set_title("FD results")
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock price")
            ax.set_zlabel("Option price diff")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            diff = a_result - a_result_zero
            ax.plot_surface(plot_x, plot_y, diff, cmap=cm.jet)
            ax.set_title("Analytical results")
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock price")
            ax.set_zlabel("Option price diff")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])
            plt.show()

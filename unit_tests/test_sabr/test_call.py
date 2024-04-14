import math
import unittest

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from models.sabr import call_option as call

plot_result = False
print_result = False


class CallOption(unittest.TestCase):
    """..."""

    def setUp(self) -> None:

        self.x_min = 2
        self.x_max = 400
        self.x_steps = 51
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        self.y_min = 0.01
        self.y_max = 1
        self.y_steps = 21
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min

        self.band = "tri"
        self.equidistant = False

        # Include time-dependent discount factor in PDE...
#        self.rate = 0.03

        self.rate = 0.0
        self.beta = 0.7
        self.vol = 0.041
        self.correlation = 0.6
        self.strike = 100

        self.t_min = 0
        self.t_max = 1
        self.t_steps = 31
        self.dt = (self.t_max - self.t_min) / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps) + self.t_min
        self.expiry_idx = self.event_grid.size - 1
        self.instrument = call.Call(self.rate, self.beta, self.vol,
                                    self.correlation, self.strike,
                                    self.expiry_idx, self.event_grid)

        self.instrument_zero = call.Call(self.rate, self.beta, self.vol,
                                         0.0, self.strike,
                                         self.expiry_idx, self.event_grid)

        if not self.equidistant:
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
            self.y_grid = y_grid[1:]

        self.instrument.fd_setup(self.x_grid, self.y_grid)
        self.instrument_zero.fd_setup(self.x_grid, self.y_grid)

    def test_1(self):
        """..."""
        # Numerical result.
        self.instrument.fd_solve()
        self.instrument_zero.fd_solve()

        # Analytical result.
        a_result = np.zeros(self.instrument.fd.solution.shape)
        a_result_zero = np.zeros(self.instrument.fd.solution.shape)
        for idx_x, x in enumerate(self.instrument.fd.grid_x):
            for idx_y, y in enumerate(self.instrument.fd.grid_y):
                a_result[idx_x, idx_y] = (
                    self.instrument.price(x, y, 0))
                a_result_zero[idx_x, idx_y] = (
                    self.instrument_zero.price(x, y, 0))

        if plot_result:
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            diff = (self.instrument.fd.solution
                    - self.instrument_zero.fd.solution)
            ax.plot_surface(plot_x, plot_y, diff, cmap=cm.jet)
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock Price")
            ax.set_zlabel("Price Difference")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            diff = a_result - a_result_zero
            ax.plot_surface(plot_x, plot_y, diff, cmap=cm.jet)
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock Price")
            ax.set_zlabel("Price Difference")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])
            plt.show()


if __name__ == '__main__':
    unittest.main()

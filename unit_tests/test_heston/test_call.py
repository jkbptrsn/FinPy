import unittest

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from models.heston import call

plot_result = False
print_result = False


class CallOption(unittest.TestCase):
    """..."""

    def setUp(self) -> None:

        self.x_min = 2
        self.x_max = 400
        self.x_steps = 101
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        self.y_min = 0
        self.y_max = 1
        self.y_steps = 21
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min

        self.band = "tri"
        self.equidistant = True

        self.rate = 0.03
        self.kappa = 3
        self.eta = 0.12
        self.vol = 0.041
        self.correlation = 0.6
        self.strike = 100

        self.t_min = 0
        self.t_max = 1
        self.t_steps = 31
        self.dt = (self.t_max - self.t_min) / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps) + self.t_min
        self.expiry_idx = self.event_grid.size - 1
        self.instrument = call.Call(self.rate, self.kappa, self.eta, self.vol,
                                    self.correlation, self.strike,
                                    self.expiry_idx, self.event_grid)

        self.instrument.fd_setup(self.x_grid, self.y_grid)

    def test_1(self):
        """..."""
        # Numerical result.
        self.instrument.fd_solve()
        # Analytical result.
        a_result = np.zeros(self.instrument.fd.solution.shape)
        for idx_x, x in enumerate(self.instrument.fd.grid_x):
            for idx_y, y in enumerate(self.instrument.fd.grid_y):
                a_result[idx_x, idx_y] = self.instrument.price(x, y, 0)

        if plot_result:
            fig = plt.figure(figsize=plt.figaspect(0.5))

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            ax.plot_surface(plot_x, plot_y, self.instrument.fd.solution, cmap=cm.jet)
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock Price")
            ax.set_zlabel("Option Price")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            diff = np.abs(self.instrument.fd.solution - a_result)
            ax.plot_surface(plot_x, plot_y, diff, cmap=cm.jet)
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock Price")
            ax.set_zlabel("Price difference")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])
            plt.pause(10)
            plt.clf()


if __name__ == '__main__':
    unittest.main()

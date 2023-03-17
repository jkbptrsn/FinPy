import math
import unittest

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import linregress

from models.heston import call
from numerics.fd.adi import craig_sneyd as cs
from numerics.fd import misc

plot_result = True
print_result = False


class CallOption(unittest.TestCase):
    """..."""

    def setUp(self) -> None:

        self.x_min = 0
        self.x_max = 400
        self.x_steps = 401
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        self.y_min = 0
        self.y_max = 1
        self.y_steps = 101
        self.dy = (self.y_max - self.y_min) / (self.y_steps - 1)
        self.y_grid = self.dy * np.arange(self.y_steps) + self.y_min

        self.band = "tri"
        self.equidistant = True

        self.rate = 0.03
        self.kappa = 3
        self.eta = 0.12
        self.vol = 0.041
        self.correlation = 0.6
        self.strike = 200

        self.t_min = 0
        self.t_max = 0.2
        self.t_steps = 401
        self.dt = (self.t_max - self.t_min) / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps) + self.t_min
        self.expiry_idx = self.event_grid.size - 1
        self.instrument = call.Call(self.rate, self.kappa, self.eta, self.vol,
                                    self.correlation, self.strike,
                                    self.expiry_idx, self.event_grid)

        self.instrument.fd_setup(self.x_grid, self.y_grid)

    def test_1(self):
        """..."""

        self.instrument.fd_solve()

        if plot_result:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_x, plot_y = np.meshgrid(self.y_grid, self.x_grid)
            ax.plot_surface(plot_x, plot_y, self.instrument.fd.solution, cmap=cm.jet)
            ax.set_xlabel("Variance")
            ax.set_ylabel("Stock Price")
            ax.set_zlabel("Option Price")
            ax.set_xlim([self.y_min, self.y_max])
            ax.set_ylim([self.x_min, self.x_max])
            plt.show()

        print(self.instrument.fd.solution[(self.x_steps - 1) // 2, (self.y_steps - 1) // 2])


if __name__ == '__main__':
    unittest.main()




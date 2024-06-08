import unittest

from matplotlib import pyplot as plt
import numpy as np

from utils import payoffs
from utils import smoothing

plot_results = False
print_results = False


class Smoothing(unittest.TestCase):

    def setUp(self) -> None:
        self.x_min = 32
        self.x_max = 70
        self.n_steps = 11
        self.dx = (self.x_max - self.x_min) / (self.n_steps - 1)
        self.x_grid = self.dx * np.arange(self.n_steps) + self.x_min

    @staticmethod
    def payoff(spot):
        return payoffs.call(spot, 50)

    def test_this(self):
        if plot_results:
            p = self.payoff(self.x_grid)
            plt.plot(self.x_grid, p, "ob", label="Payoff")
            p_smooth = smoothing.smoothing_payoff_1d(self.x_grid, self)
            plt.plot(self.x_grid, p_smooth, "or", label="Smooth payoff")
            p_linear = smoothing.smoothing_1d(self.x_grid, p)
            plt.plot(self.x_grid, p_linear, "xk", label="Linear payoff")
            plt.xlabel("Stock price")
            plt.ylabel("Payoff")
            plt.legend()
            plt.show()
        self.assertEqual(2, 2)

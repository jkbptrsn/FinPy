import unittest

import numpy as np

from models.hull_white import misc as misc_hw
from models.hull_white import caplet
from unit_tests.test_hull_white import input
from utils import misc
from utils import plots

plot_results = True
print_results = True


class Caplet(unittest.TestCase):
    """Caplet in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        self.event_steps = 401
        self.payment_date = 3
        self.dt = self.payment_date / (self.event_steps - 1)
        self.event_grid = self.dt * np.arange(self.event_steps)
        self.fixing_idx = 200
        self.payment_idx = 400

        self.cap_rate = 0.02

        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        # Caplet.
        self.time_dependence = "piecewise"
        self.caplet = caplet.CapletNew(self.kappa,
                                       self.vol,
                                       self.discount_curve,
                                       self.cap_rate,
                                       self.fixing_idx,
                                       self.payment_idx,
                                       self.event_grid,
                                       self.time_dependence)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.caplet.fd_setup(self.x_grid, equidistant=True)
        self.caplet.fd_solve()
        numerical = self.caplet.fd.solution

        analytical = self.caplet.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)

        if plot_results:
            plots.plot_price_and_greeks(self.caplet)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
#        self.assertTrue(max_error < 2.e-3)


if __name__ == '__main__':
    unittest.main()

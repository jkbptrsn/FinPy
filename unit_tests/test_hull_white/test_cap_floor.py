import unittest

import numpy as np

from models.hull_white import misc as misc_hw
from models.hull_white import cap_floor as cf_hw
from unit_tests.test_hull_white import input
from utils import misc
from utils import plots

plot_results = False
print_results = False


class CaplFloor(unittest.TestCase):
    """Cap and floor in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        self.strike_rate = 0.02

        self.event_steps = 501
        self.payment_date = 5
        self.dt = self.payment_date / (self.event_steps - 1)
        self.event_grid = self.dt * np.arange(self.event_steps)

#        self.fixing_idx = 400
#        self.payment_idx = 500
#        self.fixing_schedule = np.array([self.fixing_idx])
#        self.payment_schedule = np.array([self.payment_idx])

        self.fixing_schedule = np.array([100, 200, 300, 400])
        self.payment_schedule = np.array([200, 300, 400, 500])

        # FD spatial grid.
        self.x_min = -0.12
        self.x_max = 0.12
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        self.time_dependence = "piecewise"

        # Cap.
        self.cap = cf_hw.CapFloor(self.kappa,
                                  self.vol,
                                  self.discount_curve,
                                  self.strike_rate,
                                  self.fixing_schedule,
                                  self.payment_schedule,
                                  self.event_grid,
                                  "cap",
                                  self.time_dependence)

        # Floor.
        self.floor = cf_hw.CapFloor(self.kappa,
                                    self.vol,
                                    self.discount_curve,
                                    self.strike_rate,
                                    self.fixing_schedule,
                                    self.payment_schedule,
                                    self.event_grid,
                                    "floor",
                                    self.time_dependence)

    def test_theta_method_cap(self):
        """Finite difference pricing of cap."""
        self.cap.fd_setup(self.x_grid, equidistant=True)
        self.cap.fd_solve()
        numerical = self.cap.fd.solution
        analytical = self.cap.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.cap)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 1.e-2)

    def test_theta_method_floor(self):
        """Finite difference pricing of floor."""
        self.floor.fd_setup(self.x_grid, equidistant=True)
        self.floor.fd_solve()
        numerical = self.floor.fd.solution
        analytical = self.floor.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.floor)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.e-2)


if __name__ == '__main__':
    unittest.main()

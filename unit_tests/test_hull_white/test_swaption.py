import unittest

import numpy as np

from models.hull_white import misc as misc_hw
from models.hull_white import swap
from models.hull_white import swaption
from unit_tests.test_hull_white import input
from utils import misc
from utils import plots

plot_results = True
print_results = True


class Swaption(unittest.TestCase):
    """Payer swaption in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        self.fixed_rate = 0.02

        self.event_grid, self.fixing_schedule, self.payment_schedule = \
            misc_hw.swap_schedule(1, 5, 2, 50)

        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        # Swap.
        self.time_dependence = "piecewise"
        self.swap = swap.SwapNew(self.kappa,
                                 self.vol,
                                 self.discount_curve,
                                 self.fixed_rate,
                                 self.fixing_schedule,
                                 self.payment_schedule,
                                 self.event_grid,
                                 self.time_dependence)

        # Swaption.
        self.swaption = swaption.PayerNew(self.kappa,
                                          self.vol,
                                          self.discount_curve,
                                          self.fixed_rate,
                                          self.fixing_schedule,
                                          self.payment_schedule,
                                          self.event_grid,
                                          self.time_dependence)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.swaption.fd_setup(self.x_grid, equidistant=True)
        self.swaption.fd_solve()
        numerical = self.swaption.fd.solution
        analytical = self.swaption.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.swaption)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.e-2)


if __name__ == '__main__':
    unittest.main()

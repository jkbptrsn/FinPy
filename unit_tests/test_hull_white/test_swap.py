import unittest

import numpy as np

from models.hull_white import misc_swap as misc_sw
from models.hull_white import swap
from unit_tests.test_hull_white import input
from utils import plots

plot_results = False
print_results = False


class Swap(unittest.TestCase):
    """Fixed-for-floating swap in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        self.fixed_rate = 0.02

        self.event_grid, self.fixing_schedule, self.payment_schedule = \
            misc_sw.swap_schedule(1, 5, 2, 50)

        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        # Swap.
        self.time_dependence = "piecewise"
        self.swap = swap.Swap(self.kappa,
                              self.vol,
                              self.discount_curve,
                              self.fixed_rate,
                              self.fixing_schedule,
                              self.payment_schedule,
                              self.event_grid,
                              self.time_dependence)
        self.swap_pelsser = \
            swap.SwapPelsser(self.kappa,
                             self.vol,
                             self.discount_curve,
                             self.fixed_rate,
                             self.fixing_schedule,
                             self.payment_schedule,
                             self.event_grid,
                             self.time_dependence)

    def test_pricing(self):
        """..."""
        price_1 = self.swap.price(self.x_grid, 0)
        annuity = self.swap.annuity(self.x_grid, 0)
        forward = self.swap.par_rate(self.x_grid, 0)
        price_2 = annuity * (forward - self.fixed_rate)
        if print_results:
            for x, p1, p2 in zip(self.x_grid, price_1, price_2):
                print(x, p1, p2, p1 - p2)
        self.assertTrue(np.abs(price_1 - price_2)[(self.x_steps - 1) // 2] < 1e-12)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.swap.fd_setup(self.x_grid, equidistant=True)
        self.swap.fd_solve()
        numerical = self.swap.fd.solution
        analytical = self.swap.price(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        if plot_results:
            plots.plot_price_and_greeks(self.swap)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 6.4e-6)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of zero-coupon bond."""
        self.swap_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.swap_pelsser.fd_solve()
        numerical = self.swap_pelsser.fd.solution
        analytical = self.swap_pelsser.price(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        if plot_results:
            plots.plot_price_and_greeks(self.swap_pelsser)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 3.5e-3)


if __name__ == '__main__':
    unittest.main()

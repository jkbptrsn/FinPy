import unittest

import numpy as np

from models.hull_white import misc as misc_hw
from models.hull_white import swap
from unit_tests.test_hull_white import input
from utils import misc
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

        self.event_grid, self.fixing_schedule, self.payment_schedule = \
            misc_hw.swap_schedule(1, 10, 2, 2)

        self.fixed_rate = 0.02

        # FD spatial grid.
        self.x_min = -0.1
        self.x_max = 0.1
        self.x_steps = 11
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

    def test_pricing(self):
        """..."""
        price_1 = self.swap.price(self.x_grid, 0)
        annuity = self.swap.annuity(self.x_grid, 0)
        forward = self.swap.par_rate(self.x_grid, 0)
        price_2 = annuity * (forward - self.fixed_rate)
        if print_results:
            for x, p1, p2 in zip(self.x_grid, price_1, price_2):
                print(x, p1, p2, p1 - p2)
        self.assertTrue(np.max(np.abs(price_1 - price_2)) < 1e-12)


if __name__ == '__main__':
    unittest.main()

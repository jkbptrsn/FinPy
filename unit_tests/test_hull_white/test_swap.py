import unittest

import numpy as np

from models.hull_white import misc as misc_hw
from models.hull_white import swap
from unit_tests.test_hull_white import input
from utils import misc
from utils import plots

plot_results = True
print_results = True


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
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 51
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

    def test_int_grid(self):
        print(self.fixing_schedule)
        print(self.payment_schedule)
        print(self.event_grid)
        print(self.swap.price(self.x_grid, 0))


if __name__ == '__main__':
    unittest.main()

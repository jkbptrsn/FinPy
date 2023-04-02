import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import zero_coupon_bond as zcbond
from unit_tests.test_hull_white import input
from utils import plots

plot_results = True
print_results = True


class ZeroCouponBond(unittest.TestCase):
    """Zero-coupon bond in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        # Bond maturity.
        self.maturity = 2.5

        # FD event grid.
        self.fd_t_steps = 200
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1

#        self.time_dependence = "constant"
        self.time_dependence = "piecewise"

        # FD spatial grid.
        self.x_min = -0.1
        self.x_max = 0.25
        self.x_steps = 200
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        # Zero-coupon bond.
        self.bond = zcbond.ZCBondNew(self.kappa,
                                     self.vol,
                                     self.discount_curve,
                                     self.fd_maturity_idx,
                                     self.fd_event_grid,
                                     self.time_dependence)

    def test_theta_method_constant(self):
        """Finite difference pricing of zero-coupon bond."""
        self.bond.fd_setup(self.x_grid, equidistant=True)
        self.bond.fd_solve()
        numerical = self.bond.fd.solution
        analytical = self.bond.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
#            plt.plot(self.x_grid, relative_error, "-b")
#            plt.plot(self.x_grid - 0.027, numerical, "-r")
            plt.plot(self.x_grid, numerical, "-r")
            plt.plot(self.x_grid, analytical, "-k")
#            plt.plot(self.x_grid, self.bond.delta(self.x_grid, 0), "-r")
#            plt.plot(self.x_grid, self.bond.gamma(self.x_grid, 0), "-k")
            plt.xlabel("Short rate")
            plt.ylabel("Price/Delta/Gamma")
            plt.pause(5)

        print(self.bond.price(-0.003, 0))
        print(self.bond.delta(-0.003, 0))
        print(self.bond.gamma(-0.003, 0))


if __name__ == '__main__':
    unittest.main()

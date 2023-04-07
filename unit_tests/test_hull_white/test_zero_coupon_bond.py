import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import zero_coupon_bond as zcbond
from models.hull_white import misc as misc_hw
from unit_tests.test_hull_white import input
from utils import misc
from utils import plots

plot_results = True
print_results = True


class YFunction(unittest.TestCase):
    """Test calculation of y-function."""

    def setUp(self) -> None:
        # Speed of mean reversion strips.
        time_grid = np.array([0, 2, 4, 6, 10, 20, 30])
        kappa_grid = 0.023 * np.array([1] * 7)
        self.kappa_constant = misc.DiscreteFunc("kappa", time_grid, kappa_grid)
        # Volatility strips.
        vol_grid = 0.0165 * np.array([1] * 7)
        self.vol_constant = misc.DiscreteFunc("vol", time_grid, vol_grid)
        vol_grid = np.array([0.0165, 0.0143, 0.0140, 0.0067,
                             0.0096, 0.0087, 0.0091])
        self.vol_piecewise = misc.DiscreteFunc("vol", time_grid, vol_grid)
        # Discount curve.
        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 25
        # Event grid.
        self.t_steps = 26
        self.dt = self.maturity / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps)
        self.maturity_idx = self.t_steps - 1
        # Functions on event grid.
        self.kappa_constant_eg = \
            self.kappa_constant.interpolation(self.event_grid)
        self.vol_constant_eg = \
            self.vol_constant.interpolation(self.event_grid)
        self.vol_piecewise_eg = \
            self.vol_piecewise.interpolation(self.event_grid)
        # Bond object.
        self.bond_constant = zcbond.ZCBondNew(self.kappa_constant,
                                              self.vol_constant,
                                              self.discount_curve,
                                              self.maturity_idx,
                                              self.event_grid,
                                              "general",
                                              1 / 10)
        self.bond_piecewise = zcbond.ZCBondNew(self.kappa_constant,
                                               self.vol_piecewise,
                                               self.discount_curve,
                                               self.maturity_idx,
                                               self.event_grid,
                                               "general",
                                               1 / 200)

    def test_constant(self):
        """Test y-functions for constant vol."""
        y_constant = misc_hw.y_constant(self.kappa_constant_eg[0],
                                        self.vol_constant_eg[0],
                                        self.event_grid)
        y_piecewise = misc_hw.y_piecewise(self.kappa_constant_eg[0],
                                          self.vol_constant_eg,
                                          self.event_grid)
        y_general = self.bond_constant.y_eg
        self.assertTrue(np.max(np.abs(y_constant - y_piecewise)) < 1.e-18)
        self.assertTrue(np.max(np.abs(y_constant - y_general)) < 1.e-8)

    def test_piecewise(self):
        """Test y-functions for piecewise constant vol."""
        y_piecewise = misc_hw.y_piecewise(self.kappa_constant_eg[0],
                                          self.vol_piecewise_eg,
                                          self.event_grid)
        y_general = self.bond_piecewise.y_eg
        self.assertTrue(np.max(np.abs(y_piecewise[1:] - y_general[1:])) < 1.e6)


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
        self.fd_t_steps = 201
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

        self.int_step_factor = 3
        self.int_step_size = self.fd_dt / self.int_step_factor

        # Zero-coupon bond.
        self.bond = zcbond.ZCBondNew(self.kappa,
                                     self.vol,
                                     self.discount_curve,
                                     self.fd_maturity_idx,
                                     self.fd_event_grid,
                                     self.time_dependence,
                                     self.int_step_size)

    def test_int_grid(self):
        """Test integration grid."""
        self.assertTrue(self.bond.int_grid.size
                        == (self.fd_t_steps - 1) * self.int_step_factor + 1)

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

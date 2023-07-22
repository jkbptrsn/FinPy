import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.hull_white import misc as misc_hw
from models.hull_white import zero_coupon_bond as zcbond
from unit_tests.test_hull_white import input
from utils import data_types
from utils import plots

plot_results = False
print_results = False


class YFunction(unittest.TestCase):
    """Calculation of y-function."""

    def setUp(self) -> None:
        # Speed of mean reversion strip.
        time_grid = np.array([0, 2, 4, 6, 10, 20, 30])
        kappa_grid = 0.03 * np.ones(7)
        self.kappa_constant = \
            data_types.DiscreteFunc("kappa", time_grid, kappa_grid)
        # Volatility strips.
        vol_grid = 0.017 * np.ones(7)
        self.vol_constant = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
        vol_grid = np.array([0.017, 0.015, 0.014, 0.006, 0.009, 0.008, 0.009])
        self.vol_piecewise = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
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
        self.bond_constant = zcbond.ZCBond(self.kappa_constant,
                                           self.vol_constant,
                                           self.discount_curve,
                                           self.maturity_idx,
                                           self.event_grid,
                                           "general",
                                           1 / 10)
        self.bond_piecewise = zcbond.ZCBond(self.kappa_constant,
                                            self.vol_piecewise,
                                            self.discount_curve,
                                            self.maturity_idx,
                                            self.event_grid,
                                            "general",
                                            1 / 100)

    def test_constant(self):
        """Calculation of y-function with constant vol."""
        y_constant = misc_hw.y_constant(self.kappa_constant_eg[0],
                                        self.vol_constant_eg[0],
                                        self.event_grid)
        y_piecewise = misc_hw.y_piecewise(self.kappa_constant_eg[0],
                                          self.vol_constant_eg,
                                          self.event_grid)
        y_general = self.bond_constant.y_eg
        diff_piecewise = \
            np.abs((y_piecewise[1:] - y_constant[1:]) / y_constant[1:])
        diff_general = \
            np.abs((y_general[1:] - y_constant[1:]) / y_constant[1:])
        if print_results:
            print("y-function with constant vol.")
            print("Diff, piecewise: ", np.max(diff_piecewise))
        self.assertTrue(np.max(diff_piecewise) < 2.5e-16)
        if print_results:
            print("Diff, general  : ", np.max(diff_general))
        self.assertTrue(np.max(diff_general) < 3.0e-6)

    def test_piecewise(self):
        """Calculation of y-function with piecewise constant vol."""
        y_piecewise = misc_hw.y_piecewise(self.kappa_constant_eg[0],
                                          self.vol_piecewise_eg,
                                          self.event_grid)
        y_general = self.bond_piecewise.y_eg
        diff = np.abs((y_general[1:] - y_piecewise[1:]) / y_piecewise[1:])
        if print_results:
            print("y-function with piecewise-constant vol.")
            print("Diff: ", np.max(diff))
        self.assertTrue(np.max(diff) < 1.1e-3)


class GFunction(unittest.TestCase):
    """Calculation of G-function."""

    def setUp(self) -> None:
        # Speed of mean reversion strip.
        time_grid = np.array([0, 2, 4, 6, 10, 20, 30])
        kappa_grid = 0.03 * np.ones(7)
        self.kappa_constant = \
            data_types.DiscreteFunc("kappa", time_grid, kappa_grid)
        # Volatility strips.
        vol_grid = 0.017 * np.ones(7)
        self.vol_constant = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
        # Discount curve.
        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 25
        # Event grid.
        self.t_steps = 26
        self.dt = self.maturity / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps)
        self.maturity_idx = self.t_steps - 1
        # Function on event grid.
        self.kappa_constant_eg = \
            self.kappa_constant.interpolation(self.event_grid)
        # Bond object.
        self.bond_constant = zcbond.ZCBond(self.kappa_constant,
                                           self.vol_constant,
                                           self.discount_curve,
                                           self.maturity_idx,
                                           self.event_grid,
                                           "general",
                                           1 / 10)

    def test_constant(self):
        """Calculation of G-function with constant kappa."""
        g_constant = misc_hw.g_constant(self.kappa_constant_eg[0],
                                        self.event_grid)
        g_constant = misc_hw.g_function(self.maturity_idx, g_constant,
                                        self.bond_constant.int_kappa_eg)
        g_general = self.bond_constant.gt_eg
        diff = np.abs((g_constant[:-1] - g_general[:-1]) / g_constant[:-1])
        if plot_results:
            plt.plot(self.event_grid, g_constant, "-b")
            plt.plot(self.event_grid, g_general, "or")
            plt.xlabel("t")
            plt.ylabel("G(t,t_maturity)")
            plt.show()
        if print_results:
            print("G-function with constant kappa.")
            print(np.max(diff))
        self.assertTrue(np.max(diff) < 7.5e-7)


class AlphaFunction(unittest.TestCase):
    """Calculation of alpha-function."""

    def setUp(self) -> None:
        # Speed of mean reversion strips.
        time_grid = np.array([0, 2, 4, 6, 10, 20, 30])
        kappa_grid = 0.023 * np.array([1] * 7)
        self.kappa_constant = \
            data_types.DiscreteFunc("kappa", time_grid, kappa_grid)
        # Volatility strips.
        vol_grid = 0.0165 * np.array([1] * 7)
        self.vol_constant = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
        vol_grid = np.array([0.0165, 0.0143, 0.0140, 0.0067,
                             0.0096, 0.0087, 0.0091])
        self.vol_piecewise = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
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
        self.bond_constant = zcbond.ZCBond(self.kappa_constant,
                                           self.vol_constant,
                                           self.discount_curve,
                                           self.maturity_idx,
                                           self.event_grid,
                                           "general",
                                           1 / 10)
        self.bond_piecewise = zcbond.ZCBond(self.kappa_constant,
                                            self.vol_piecewise,
                                            self.discount_curve,
                                            self.maturity_idx,
                                            self.event_grid,
                                            "general",
                                            1 / 100)

    def test_constant(self):
        """Calculation of alpha-function with constant vol."""
        alpha_constant = \
            misc_hw.alpha_constant(self.kappa_constant_eg[0],
                                   self.vol_constant_eg[0],
                                   self.event_grid)
        alpha_piecewise = \
            misc_hw.alpha_piecewise(self.kappa_constant_eg[0],
                                    self.vol_constant_eg,
                                    self.event_grid)
        alpha_general = \
            misc_hw.alpha_general(self.bond_constant.int_grid,
                                  self.bond_constant.int_event_idx,
                                  self.bond_constant.int_kappa_step_ig,
                                  self.bond_constant.vol_ig,
                                  self.bond_constant.event_grid)
        diff_piecewise = np.abs((alpha_piecewise[1:]
                                 - alpha_constant[1:]) / alpha_constant[1:])
        diff_general = np.abs((alpha_general[1:]
                               - alpha_constant[1:]) / alpha_constant[1:])
        self.assertTrue(np.max(diff_piecewise) < 8.e-13)
        self.assertTrue(np.max(diff_general) < 3.e-6)

    def test_piecewise(self):
        """Calculation of alpha-function with piecewise constant vol."""
        alpha_piecewise = \
            misc_hw.alpha_piecewise(self.kappa_constant_eg[0],
                                    self.vol_piecewise_eg,
                                    self.event_grid)
        alpha_general = \
            misc_hw.alpha_general(self.bond_piecewise.int_grid,
                                  self.bond_piecewise.int_event_idx,
                                  self.bond_piecewise.int_kappa_step_ig,
                                  self.bond_piecewise.vol_ig,
                                  self.bond_piecewise.event_grid)
        diff = np.abs((alpha_general[1:]
                       - alpha_piecewise[1:]) / alpha_piecewise[1:])
        self.assertTrue(np.max(diff) < 7.e-4)


class IntAlphaFunction(unittest.TestCase):
    """Calculation of integral of alpha-function."""

    def setUp(self) -> None:
        # Speed of mean reversion strips.
        time_grid = np.array([0, 2, 4, 6, 10, 20, 30])
        kappa_grid = 0.023 * np.array([1] * 7)
        self.kappa_constant = \
            data_types.DiscreteFunc("kappa", time_grid, kappa_grid)
        # Volatility strips.
        vol_grid = 0.0165 * np.array([1] * 7)
        self.vol_constant = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
        vol_grid = np.array([0.0165, 0.0143, 0.0140, 0.0067,
                             0.0096, 0.0087, 0.0091])
        self.vol_piecewise = \
            data_types.DiscreteFunc("vol", time_grid, vol_grid)
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
        self.bond_constant = zcbond.ZCBond(self.kappa_constant,
                                           self.vol_constant,
                                           self.discount_curve,
                                           self.maturity_idx,
                                           self.event_grid,
                                           "general",
                                           1 / 100)
        self.bond_piecewise = zcbond.ZCBond(self.kappa_constant,
                                            self.vol_piecewise,
                                            self.discount_curve,
                                            self.maturity_idx,
                                            self.event_grid,
                                            "general",
                                            1 / 100)

    def test_constant(self):
        """Calculation of integral alpha-function with constant vol."""
        int_alpha_constant = \
            misc_hw.int_alpha_constant(self.kappa_constant_eg[0],
                                       self.vol_constant_eg[0],
                                       self.event_grid)
        int_alpha_piecewise = \
            misc_hw.int_alpha_piecewise(self.kappa_constant_eg[0],
                                        self.vol_constant_eg,
                                        self.event_grid)
        int_alpha_general = \
            misc_hw.int_alpha_general(self.bond_constant.int_grid,
                                      self.bond_constant.int_event_idx,
                                      self.bond_constant.int_kappa_step_ig,
                                      self.bond_constant.vol_ig,
                                      self.bond_constant.event_grid)
        diff_piecewise = \
            np.abs((int_alpha_piecewise[1:]
                    - int_alpha_constant[1:]) / int_alpha_constant[1:])
        diff_general = \
            np.abs((int_alpha_general[1:]
                    - int_alpha_constant[1:]) / int_alpha_constant[1:])
        self.assertTrue(np.max(diff_piecewise) < 2.e-11)
        self.assertTrue(np.max(diff_general) < 5.e-3)

    def test_piecewise(self):
        """Calculation of integral of alpha-function with piecewise constant vol."""
        int_alpha_piecewise = \
            misc_hw.int_alpha_piecewise(self.kappa_constant_eg[0],
                                        self.vol_piecewise_eg,
                                        self.event_grid)
        int_alpha_general = \
            misc_hw.int_alpha_general(self.bond_piecewise.int_grid,
                                      self.bond_piecewise.int_event_idx,
                                      self.bond_piecewise.int_kappa_step_ig,
                                      self.bond_piecewise.vol_ig,
                                      self.bond_piecewise.event_grid)
        diff = np.abs((int_alpha_general[1:]
                       - int_alpha_piecewise[1:]) / int_alpha_piecewise[1:])
        self.assertTrue(np.max(diff) < 6.e-3)


class ZeroCouponBond(unittest.TestCase):
    """Zero-coupon bond in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip

#        self.vol = input.vol_strip_constant
        self.vol = input.vol_strip

        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 30
        # FD event grid.
        self.fd_t_steps = 361
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 301
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.int_step_factor = 50  # 200 # 3
        self.int_step_size = self.fd_dt / self.int_step_factor
        # Zero-coupon bond.

#        self.time_dependence = "constant"
        self.time_dependence = "piecewise"
#        self.time_dependence = "general"

        self.bond = zcbond.ZCBond(self.kappa,
                                  self.vol,
                                  self.discount_curve,
                                  self.fd_maturity_idx,
                                  self.fd_event_grid,
                                  self.time_dependence,
                                  self.int_step_size)

        self.bond_pelsser = \
            zcbond.ZCBondPelsser(self.kappa,
                                 self.vol,
                                 self.discount_curve,
                                 self.fd_maturity_idx,
                                 self.fd_event_grid,
                                 self.time_dependence,
                                 self.int_step_size)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.bond.fd_setup(self.x_grid, equidistant=True)
        self.bond.fd_solve()
        numerical = self.bond.fd.solution
        analytical = self.bond.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.bond)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.e-3)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of zero-coupon bond."""
        print(self.bond_pelsser.transformation)
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.bond_pelsser.fd_solve()
        numerical = self.bond_pelsser.fd.solution
        analytical = self.bond_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.bond_pelsser)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.e-3)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of zero-coupon bond."""
        x_grid = 0.02 * np.arange(11) - 0.1
        analytical = self.bond.price(x_grid, 0)
        n_paths = 30000
        self.bond.mc_exact_setup(time_dependence="piecewise")
        numerical = np.zeros(x_grid.size)
        error = np.zeros(x_grid.size)
        rng = np.random.default_rng(0)
        for idx, spot in enumerate(x_grid):
            self.bond.mc_exact_solve(spot, n_paths, rng=rng, antithetic=True)
            numerical[idx] = self.bond.mc_exact.solution
            error[idx] = self.bond.mc_exact.error
        if plot_results:
            plt.plot(x_grid, analytical, "-b")
            plt.errorbar(x_grid, numerical, yerr=error, fmt="or")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Zero-coupon bond price")
            plt.show()
        relative_error = np.abs((analytical - numerical) / analytical)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 5e-3)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of zero-coupon bond."""
        x_grid = 0.02 * np.arange(11) - 0.1
        analytical = self.bond_pelsser.price(x_grid, 0)
        n_paths = 30000
        self.bond_pelsser.mc_exact_setup(time_dependence="piecewise")
        numerical = np.zeros(x_grid.size)
        error = np.zeros(x_grid.size)
        rng = np.random.default_rng(0)
        for idx, spot in enumerate(x_grid):
            self.bond_pelsser.mc_exact_solve(spot, n_paths, rng=rng, antithetic=True)
            numerical[idx] = self.bond_pelsser.mc_exact.solution
            error[idx] = self.bond_pelsser.mc_exact.error
        if plot_results:
            plt.plot(x_grid, analytical, "-b")
            plt.errorbar(x_grid, numerical, yerr=error, fmt="or")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Zero-coupon bond price")
            plt.show()
        relative_error = np.abs((analytical - numerical) / analytical)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 5e-3)


if __name__ == '__main__':
    unittest.main()

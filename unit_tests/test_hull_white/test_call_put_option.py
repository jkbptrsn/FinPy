import unittest

import numpy as np

from models.hull_white import european_option as option
from models.hull_white import misc as misc_hw
from unit_tests.test_hull_white import input
from utils import data_types
from utils import plots

plot_results = True
print_results = True


class VFunction(unittest.TestCase):
    """Calculation of v-function."""

    def setUp(self) -> None:
        # Speed of mean reversion strip.
        time_grid = np.array([0, 2, 4, 6, 10, 20, 30])
        kappa_grid = 0.023 * np.array([1] * 7)
        self.kappa_constant = \
            data_types.DiscreteFunc("kappa", time_grid, kappa_grid)
        # Volatility strip.
        vol_grid = 0.0165 * np.array([1] * 7)
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
        # Option expiry.
        self.expiry_idx = (self.t_steps - 1) // 2 + 1
        self.expiry = self.event_grid[self.expiry_idx]
        # Option strike price.
        self.strike = 0.8
        # Functions on event grid. TODO: Why interpolation? Use kappa_constant...
        self.kappa_constant_eg = \
            self.kappa_constant.interpolation(self.event_grid)
        self.vol_constant_eg = \
            self.vol_constant.interpolation(self.event_grid)
        # Call object.
        self.call = option.EuropeanOption(self.kappa_constant,
                                        self.vol_constant,
                                        self.discount_curve,
                                        self.strike,
                                        self.expiry_idx,
                                        self.maturity_idx,
                                        self.event_grid,
                                        "piecewise",
                                        1 / 100)

    def test_constant(self):
        """Calculation of v-function with constant vol."""
        v_constant = misc_hw.v_constant(self.kappa_constant_eg[0],
                                        self.vol_constant_eg[0],
                                        self.expiry_idx,
                                        self.maturity_idx,
                                        self.call.zcbond.g_eg,
                                        self.event_grid)
        v_piecewise = self.call.v_eg
        print(np.max(np.abs(v_constant - v_piecewise)))
        self.assertTrue(np.max(np.abs(v_constant - v_piecewise)) < 5.e-3)


class Call(unittest.TestCase):
    """European call option in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 30
        # FD event grid.
        self.fd_t_steps = 361
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1
        # Option expiry.
        self.fd_expiry_idx = (self.fd_t_steps - 1) // 2 + 1
        self.expiry = self.fd_event_grid[self.fd_expiry_idx]
        # Option strike price.
        self.strike = 0.8
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.int_step_factor = 3
        self.int_step_size = self.fd_dt / self.int_step_factor
        # Call option.
        self.time_dependence = "piecewise"
        self.call = option.EuropeanOption(self.kappa,
                                        self.vol,
                                        self.discount_curve,
                                        self.strike,
                                        self.fd_expiry_idx,
                                        self.fd_maturity_idx,
                                        self.fd_event_grid,
                                        self.time_dependence,
                                        self.int_step_size)

        self.callPelsser = option.EuropeanOptionPelsser(self.kappa,
                                                      self.vol,
                                                      self.discount_curve,
                                                      self.strike,
                                                      self.fd_expiry_idx,
                                                      self.fd_maturity_idx,
                                                      self.fd_event_grid,
                                                      self.time_dependence,
                                                      self.int_step_size)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.call.fd_setup(self.x_grid, equidistant=True)
        self.call.fd_solve()
        numerical = self.call.fd.solution
        analytical = self.call.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.call)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.002)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.002)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 9.8-3)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of zero-coupon bond."""
        self.callPelsser.fd_setup(self.x_grid, equidistant=True)
        self.callPelsser.fd_solve()
        numerical = self.callPelsser.fd.solution
        analytical = self.callPelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.callPelsser)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.002)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.002)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 8.e-3)


class Put(unittest.TestCase):
    """European put option in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 30
        # FD event grid.
        self.fd_t_steps = 361
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1
        # Option expiry.
        self.fd_expiry_idx = (self.fd_t_steps - 1) // 2 + 1
        self.expiry = self.fd_event_grid[self.fd_expiry_idx]
        # Option strike price.
        self.strike = 0.8
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.int_step_factor = 3
        self.int_step_size = self.fd_dt / self.int_step_factor
        # Put option.
        self.time_dependence = "piecewise"
#        self.put = put.Put(self.kappa,
#                           self.vol,
#                           self.discount_curve,
#                           self.strike,
#                           self.fd_expiry_idx,
#                           self.fd_maturity_idx,
#                           self.fd_event_grid,
#                           self.time_dependence,
#                           self.int_step_size)
        self.put = option.EuropeanOption(self.kappa,
                                         self.vol,
                                         self.discount_curve,
                                         self.strike,
                                         self.fd_expiry_idx,
                                         self.fd_maturity_idx,
                                         self.fd_event_grid,
                                         self.time_dependence,
                                         self.int_step_size,
                                         "Put")

        self.putPelsser = option.EuropeanOptionPelsser(self.kappa,
                                                       self.vol,
                                                       self.discount_curve,
                                                       self.strike,
                                                       self.fd_expiry_idx,
                                                       self.fd_maturity_idx,
                                                       self.fd_event_grid,
                                                       self.time_dependence,
                                                       self.int_step_size,
                                                       "Put")

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.put.fd_setup(self.x_grid, equidistant=True)
        self.put.fd_solve()
        numerical = self.put.fd.solution
        analytical = self.put.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.put)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.002)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.002)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.1e-3)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of zero-coupon bond."""
        self.putPelsser.fd_setup(self.x_grid, equidistant=True)
        self.putPelsser.fd_solve()
        numerical = self.putPelsser.fd.solution
        analytical = self.putPelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.putPelsser)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.002)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.002)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 7.e-4)


if __name__ == '__main__':
    unittest.main()

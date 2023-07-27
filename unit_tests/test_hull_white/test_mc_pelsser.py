import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import mc_andersen as sde
from models.hull_white import misc as misc_hw
from utils import data_types

plot_results = False
print_results = False


class Misc(unittest.TestCase):
    """Various functions used in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Event dates in year fractions.
        self.event_grid = np.arange(16)
        # Speed of mean reversion strip.
        self.kappa_scalar = 0.02
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = data_types.DiscreteFunc("kappa1",
                                              self.event_grid,
                                              self.kappa_vector1)
        # Volatility strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        # Constant vol strip.
        self.vol1 = \
            data_types.DiscreteFunc("vol1", self.event_grid, self.vol_vector1)
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            self.vol_vector2[idx] = (idx % 4 + 1) * self.vol_vector1[idx]
        # Piecewise-constant vol strip.
        self.vol2 = \
            data_types.DiscreteFunc("vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = \
            data_types.DiscreteFunc("discount", self.event_grid,
                                    np.ones(self.event_grid.size))
        # SDE object, constant vol strip.
        self.sde1 = sde.SdeExactGeneral(self.kappa1,
                                        self.vol1,
                                        self.discount_curve,
                                        self.event_grid,
                                        int_dt=1 / 100)
        # SDE object, piecewise-constant vol strip.
        self.sde2 = sde.SdeExactGeneral(self.kappa1,
                                        self.vol2,
                                        self.discount_curve,
                                        self.event_grid,
                                        int_dt=1 / 200)

    def test_alpha_constant(self):
        """Alpha-function with constant vol."""
        alpha_constant = \
            misc_hw.alpha_constant(self.kappa_scalar,
                                   self.vol_scalar,
                                   self.event_grid)
        alpha_piecewise = \
            misc_hw.alpha_piecewise(self.kappa_scalar,
                                    self.vol_vector1,
                                    self.event_grid)
        alpha_general = \
            misc_hw.alpha_general(self.sde1.int_grid,
                                  self.sde1.int_event_idx,
                                  self.sde1.int_kappa_step_ig,
                                  self.sde1.vol_ig,
                                  self.sde1.event_grid)
        diff_piecewise = np.abs((alpha_piecewise[1:]
                                 - alpha_constant[1:]) / alpha_constant[1:])
        diff_general = np.abs((alpha_general[1:]
                               - alpha_constant[1:]) / alpha_constant[1:])
        if plot_results:
            plt.plot(self.event_grid, alpha_constant, "-b", label="Constant")
            plt.plot(self.event_grid, alpha_piecewise, "or", label="Piecewise")
            plt.plot(self.event_grid, alpha_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("alpha-function")
            plt.legend()
            plt.show()
        if print_results:
            print(diff_piecewise)
            print(diff_general)
        self.assertTrue(np.max(diff_piecewise) < 1.5e-13)
        self.assertTrue(np.max(diff_general) < 1.7e-8)

    def test_alpha_piecewise(self):
        """Alpha-function with piecewise constant vol."""
        alpha_piecewise = \
            misc_hw.alpha_piecewise(self.kappa_scalar,
                                    self.vol_vector2,
                                    self.event_grid)
        alpha_general = \
            misc_hw.alpha_general(self.sde2.int_grid,
                                  self.sde2.int_event_idx,
                                  self.sde2.int_kappa_step_ig,
                                  self.sde2.vol_ig,
                                  self.sde2.event_grid)
        diff = np.abs((alpha_general[1:]
                       - alpha_piecewise[1:]) / alpha_piecewise[1:])
        if plot_results:
            plt.plot(self.event_grid, alpha_piecewise, "or", label="Piecewise")
            plt.plot(self.event_grid, alpha_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("alpha-function")
            plt.legend()
            plt.show()
        if print_results:
            print(diff)
        self.assertTrue(np.max(diff) < 2.2e-3)

    def test_int_alpha_constant(self):
        """Integral of alpha-function with constant vol."""
        int_alpha_constant = \
            misc_hw.int_alpha_constant(self.kappa_scalar,
                                       self.vol_scalar,
                                       self.event_grid)
        int_alpha_piecewise = \
            misc_hw.int_alpha_piecewise(self.kappa_scalar,
                                        self.vol_vector1,
                                        self.event_grid)
        int_alpha_general = \
            misc_hw.int_alpha_general(self.sde1.int_grid,
                                      self.sde1.int_event_idx,
                                      self.sde1.int_kappa_step_ig,
                                      self.sde1.vol_ig,
                                      self.sde1.event_grid)
        diff_piecewise = \
            np.abs((int_alpha_piecewise[1:]
                    - int_alpha_constant[1:]) / int_alpha_constant[1:])
        diff_general = \
            np.abs((int_alpha_general[1:]
                    - int_alpha_constant[1:]) / int_alpha_constant[1:])
        if plot_results:
            plt.plot(self.event_grid, int_alpha_constant,
                     "-b", label="Constant")
            plt.plot(self.event_grid, int_alpha_piecewise,
                     "or", label="Piecewise")
            plt.plot(self.event_grid, int_alpha_general,
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("int_alpha-function")
            plt.legend()
            plt.show()
        if print_results:
            print(diff_piecewise)
            print(diff_general)
        self.assertTrue(np.max(diff_piecewise) < 9.0e-11)
        self.assertTrue(np.max(diff_general) < 4.8e-3)

    def test_int_alpha_piecewise(self):
        """Integral of alpha-function with piecewise constant vol."""
        int_alpha_piecewise = \
            misc_hw.int_alpha_piecewise(self.kappa_scalar,
                                        self.vol_vector2,
                                        self.event_grid)
        int_alpha_general = \
            misc_hw.int_alpha_general(self.sde2.int_grid,
                                      self.sde2.int_event_idx,
                                      self.sde2.int_kappa_step_ig,
                                      self.sde2.vol_ig,
                                      self.sde2.event_grid)
        diff = np.abs((int_alpha_general[1:]
                       - int_alpha_piecewise[1:]) / int_alpha_piecewise[1:])
        if plot_results:
            plt.plot(self.event_grid, int_alpha_piecewise,
                     "or", label="Piecewise")
            plt.plot(self.event_grid, int_alpha_general,
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("int_alpha-function")
            plt.legend()
            plt.show()
        if print_results:
            print(diff)
        self.assertTrue(np.max(diff) < 2.1e-3)


class SDE(unittest.TestCase):
    """SDE classes in 1-factor Hull-White model.

    TODO: Change to pelsser SDE!
    """

    def setUp(self) -> None:
        # Event dates in year fractions.
        self.event_grid = np.arange(16)
        # Speed of mean reversion strip.
        self.kappa_scalar = 0.15
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = data_types.DiscreteFunc("kappa1", self.event_grid,
                                              self.kappa_vector1)
        # Volatility strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        # Constant vol strip.
        self.vol1 = \
            data_types.DiscreteFunc("vol1", self.event_grid, self.vol_vector1)
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            if idx % 2 == 0:
                self.vol_vector2[idx] = self.vol_vector1[idx]
            else:
                self.vol_vector2[idx] = 2 * self.vol_vector1[idx]
        # Piecewise-constant vol strip.
        self.vol2 = \
            data_types.DiscreteFunc("vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = \
            data_types.DiscreteFunc("discount", self.event_grid,
                                    np.ones(self.event_grid.size))
        # SDE objects.
        self.sde_constant = sde.SdeExactConstant(self.kappa1,
                                                 self.vol1,
                                                 self.discount_curve,
                                                 self.event_grid)
        self.sde_piecewise1 = sde.SdeExactPiecewise(self.kappa1,
                                                    self.vol1,
                                                    self.discount_curve,
                                                    self.event_grid)
        self.sde_piecewise2 = sde.SdeExactPiecewise(self.kappa1,
                                                    self.vol2,
                                                    self.discount_curve,
                                                    self.event_grid)
        self.sde_general1 = sde.SdeExactGeneral(self.kappa1,
                                                self.vol1,
                                                self.discount_curve,
                                                self.event_grid,
                                                int_dt=1 / 50)
        self.sde_general2 = sde.SdeExactGeneral(self.kappa1,
                                                self.vol2,
                                                self.discount_curve,
                                                self.event_grid,
                                                int_dt=1 / 50)

    def test_sde_constant_vol(self):
        """Test SDE classes for constant vol-strip."""
        # Number of Monte-Carlo paths.
        n_paths = 200000
        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve.values
        # SDE constant.
        self.sde_constant.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_constant.discount_adjustment(
            self.sde_constant.discount_paths,
            self.sde_constant.discount_curve_eg)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE constant: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 8.1e-4)
        # SDE piecewise.
        self.sde_piecewise1.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_piecewise1.discount_adjustment(
            self.sde_piecewise1.discount_paths,
            self.sde_piecewise1.discount_curve_eg)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE piecewise: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 8.1e-4)
        # SDE general.
        self.sde_general1.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_general1.discount_adjustment(
            self.sde_general1.discount_paths,
            self.sde_general1.discount_curve_eg)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE general: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 8.10e-3)

        if plot_results:
            plt.plot(self.event_grid, self.sde_constant.y_eg,
                     "-b", label="Constant")
            plt.plot(self.event_grid, self.sde_piecewise1.y_eg,
                     "or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general1.y_eg,
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("y-function")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_constant.rate_mean[:, 0],
                     "-b", label="Constant")
            plt.plot(self.event_grid, self.sde_piecewise1.rate_mean[:, 0],
                     "or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general1.rate_mean[:, 0],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Rate mean, 1")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_constant.rate_mean[:, 1],
                     "-b", label="Constant")
            plt.plot(self.event_grid, self.sde_piecewise1.rate_mean[:, 1],
                     "or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general1.rate_mean[:, 1],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Rate mean, 2")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_constant.discount_mean[:, 0],
                     "-b", label="Constant")
            plt.plot(self.event_grid, self.sde_piecewise1.discount_mean[:, 0],
                     "or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general1.discount_mean[:, 0],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Discount mean, 1")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_constant.discount_mean[:, 1],
                     "-b", label="Constant")
            plt.plot(self.event_grid, self.sde_piecewise1.discount_mean[:, 1],
                     "or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general1.discount_mean[:, 1],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Discount mean, 2")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_constant.covariance,
                     "-b", label="Constant")
            plt.plot(self.event_grid, self.sde_piecewise1.covariance,
                     "or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general1.covariance,
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Covariance")
            plt.legend()
            plt.show()

    def test_sde_piecewise_vol(self):
        """Test SDE classes for piecewise-constant vol-strip."""
        # Number of Monte-Carlo paths.
        n_paths = 200000
        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve.values
        # SDE piecewise.
        self.sde_piecewise2.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_piecewise2.discount_adjustment(
            self.sde_piecewise2.discount_paths,
            self.sde_piecewise2.discount_curve_eg)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE piecewise: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 1.8e-3)
        # SDE general.
        self.sde_general2.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_general2.discount_adjustment(
            self.sde_general2.discount_paths,
            self.sde_general2.discount_curve_eg)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE general: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 1.8e-3)

        if plot_results:
            plt.plot(self.event_grid, self.sde_piecewise2.y_eg,
                     "-or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general2.y_eg,
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("y-function")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_piecewise2.rate_mean[:, 0],
                     "-or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general2.rate_mean[:, 0],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Rate mean, 1")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_piecewise2.rate_mean[:, 1],
                     "-or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general2.rate_mean[:, 1],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Rate mean, 2")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_piecewise2.discount_mean[:, 0],
                     "-or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general2.discount_mean[:, 0],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Discount mean, 1")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_piecewise2.discount_mean[:, 1],
                     "-or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general2.discount_mean[:, 1],
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Discount mean, 2")
            plt.legend()
            plt.show()

            plt.plot(self.event_grid, self.sde_piecewise2.covariance,
                     "-or", label="Piecewise")
            plt.plot(self.event_grid, self.sde_general2.covariance,
                     "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Covariance")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    unittest.main()

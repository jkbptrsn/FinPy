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
        self.kappa_scalar = 0.15
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = \
            data_types.DiscreteFunc("kappa1", self.event_grid, self.kappa_vector1)
        # Volatility strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        self.vol1 = \
            data_types.DiscreteFunc("vol1", self.event_grid, self.vol_vector1)
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            if idx % 2 == 0:
                self.vol_vector2[idx] = self.vol_vector1[idx]
            else:
                self.vol_vector2[idx] = 4 * self.vol_vector1[idx]
        self.vol2 = \
            data_types.DiscreteFunc("vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = \
            data_types.DiscreteFunc("discount", self.event_grid,
                                    np.ones(self.event_grid.size))
        # SDE objects.
        self.sde1 = sde.SdeExactGeneral(self.kappa1, self.vol1,
                                        self.discount_curve, self.event_grid,
                                        int_dt=1 / 12)
        self.sde2 = sde.SdeExactGeneral(self.kappa1, self.vol2,
                                        self.discount_curve, self.event_grid,
                                        int_dt=1 / 100)

    def test_y_constant(self):
        """Numerical evaluation of y-function."""
        y_constant = misc_hw.y_constant(self.kappa_scalar, self.vol_scalar,
                                        self.event_grid)
        y_piecewise = misc_hw.y_piecewise(self.kappa_scalar, self.vol_vector1,
                                          self.event_grid)
        y_general, _ = misc_hw.y_general(self.sde1.int_grid,
                                         self.sde1.int_event_idx,
                                         self.sde1.int_kappa_step_ig,
                                         self.sde1.vol_ig,
                                         self.sde1.event_grid)
        if plot_results:
            plt.plot(self.event_grid, y_constant, "-b", label="Constant")
            plt.plot(self.event_grid, y_piecewise, "or", label="Piecewise")
            plt.plot(self.event_grid, y_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("y-function")
            plt.legend()
            plt.show()
        for idx, (y1, y2, y3) in \
                enumerate(zip(y_constant, y_piecewise, y_general)):
            if idx > 0:
                diff1 = abs(y1 - y2) / y1
                diff2 = abs(y1 - y3) / y1
                if print_results:
                    print(diff1, diff2)
                self.assertTrue(diff1 < 1.0e-15)
                self.assertTrue(diff2 < 6.0e-5)

    def test_y_piecewise(self):
        """Numerical evaluation of y-function."""
        y_piecewise = misc_hw.y_piecewise(self.kappa_scalar, self.vol_vector2,
                                          self.event_grid)
        y_general, _ = misc_hw.y_general(self.sde2.int_grid,
                                         self.sde2.int_event_idx,
                                         self.sde2.int_kappa_step_ig,
                                         self.sde2.vol_ig,
                                         self.sde2.event_grid)
        if plot_results:
            plt.plot(self.event_grid, y_piecewise, "-or", label="Piecewise")
            plt.plot(self.event_grid, y_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("y-function")
            plt.legend()
            plt.show()
        for idx, (y1, y2) in enumerate(zip(y_piecewise, y_general)):
            if idx > 1:
                diff = abs(y1 - y2) / y1
                if print_results:
                    print(y1, y2, diff)
                self.assertTrue(diff < 6.0e-3)

    def test_int_y_constant(self):
        """Numerical evaluation of "integral" of y-function."""
        y_constant = \
            misc_hw.int_y_constant(self.kappa_scalar, self.vol_scalar,
                                   self.event_grid)
        y_piecewise = \
            misc_hw.int_y_piecewise(self.kappa_scalar, self.vol_vector1,
                                    self.event_grid)
        y_general = misc_hw.int_y_general(self.sde1.int_grid,
                                          self.sde1.int_event_idx,
                                          self.sde1.int_kappa_step_ig,
                                          self.sde1.vol_ig,
                                          self.sde1.event_grid)
        if plot_results:
            plt.plot(self.event_grid, y_constant, "-b", label="Constant")
            plt.plot(self.event_grid, y_piecewise, "or", label="Piecewise")
            plt.plot(self.event_grid, y_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Integral of y-function")
            plt.legend()
            plt.show()
        for idx, (y1, y2, y3) in \
                enumerate(zip(y_constant, y_piecewise, y_general)):
            if idx > 0:
                diff1 = abs(y1 - y2) / y1
                diff2 = abs(y1 - y3) / y1
                if print_results:
                    print(diff1, diff2)
                self.assertTrue(diff1 < 5.0e-15)
                self.assertTrue(diff2 < 7.0e-5)

    def test_int_y_piecewise(self):
        """Test numerical evaluation of "integral" of y-function."""
        y_piecewise = \
            misc_hw.int_y_piecewise(self.kappa_scalar, self.vol_vector2,
                                    self.event_grid)
        y_general = misc_hw.int_y_general(self.sde2.int_grid,
                                          self.sde2.int_event_idx,
                                          self.sde2.int_kappa_step_ig,
                                          self.sde2.vol_ig,
                                          self.sde2.event_grid)
        if plot_results:
            plt.plot(self.event_grid, y_piecewise, "-or", label="Piecewise")
            plt.plot(self.event_grid, y_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Integral of y-function")
            plt.legend()
            plt.show()
        for idx, (y1, y2) in enumerate(zip(y_piecewise, y_general)):
            if idx > 1:
                diff = abs(y1 - y2) / y1
                if print_results:
                    print(y1, y2, diff)
                self.assertTrue(diff < 8.0e-3)

    def test_double_int_y_constant(self):
        """Numerical evaluation of "double integral" of y-function."""
        y_constant = \
            misc_hw.double_int_y_constant(self.kappa_scalar, self.vol_scalar,
                                          self.event_grid)
        y_piecewise = \
            misc_hw.double_int_y_piecewise(self.kappa_scalar, self.vol_vector1,
                                           self.event_grid)
        y_general = misc_hw.double_int_y_general(self.sde1.int_grid,
                                                 self.sde1.int_event_idx,
                                                 self.sde1.int_kappa_step_ig,
                                                 self.sde1.vol_ig,
                                                 self.sde1.event_grid)
        if plot_results:
            plt.plot(self.event_grid, y_constant, "-b", label="Constant")
            plt.plot(self.event_grid, y_piecewise, "or", label="Piecewise")
            plt.plot(self.event_grid, y_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Double integral of y-function")
            plt.legend()
            plt.show()
        for idx, (y1, y2, y3) in \
                enumerate(zip(y_constant, y_piecewise, y_general)):
            if idx > 0:
                diff1 = abs(y1 - y2) / y1
                diff2 = abs(y1 - y3) / y1
                if print_results:
                    print(diff1, diff2)
                self.assertTrue(diff1 < 2.0e-13)
                self.assertTrue(diff2 < 4.0e-3)

    def test_double_int_y_piecewise(self):
        """Test numerical evaluation of "double integral" of y-function."""
        y_piecewise = \
            misc_hw.double_int_y_piecewise(self.kappa_scalar, self.vol_vector2,
                                           self.event_grid)
        y_general = misc_hw.double_int_y_general(self.sde2.int_grid,
                                                 self.sde2.int_event_idx,
                                                 self.sde2.int_kappa_step_ig,
                                                 self.sde2.vol_ig,
                                                 self.sde2.event_grid)
        if plot_results:
            plt.plot(self.event_grid, y_piecewise, "-or", label="Piecewise")
            plt.plot(self.event_grid, y_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("Double integral of y-function")
            plt.legend()
            plt.show()
        for idx, (y1, y2) in enumerate(zip(y_piecewise, y_general)):
            if idx > 2:
                diff = abs(y1 - y2) / y1
                if print_results:
                    print(y1, y2, diff)
                self.assertTrue(diff < 4.0e-3)


class SDE(unittest.TestCase):
    """SDE classes in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Event dates in year fractions.
        self.event_grid = np.arange(16)
        # Speed of mean reversion strip.
        self.kappa_scalar = 0.15
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = \
            data_types.DiscreteFunc("kappa1", self.event_grid,
                                    self.kappa_vector1)
        # Volatility strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        self.vol1 = \
            data_types.DiscreteFunc("vol1", self.event_grid, self.vol_vector1)
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            if idx % 2 == 0:
                self.vol_vector2[idx] = self.vol_vector1[idx]
            else:
                self.vol_vector2[idx] = 2 * self.vol_vector1[idx]
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
                                                int_dt=1 / 100)
        self.sde_general2 = sde.SdeExactGeneral(self.kappa1,
                                                self.vol2,
                                                self.discount_curve,
                                                self.event_grid,
                                                int_dt=1 / 100)

    def test_sde_constant_vol(self):
        """Test SDE classes for constant vol-strip."""
        # Number of Monte-Carlo paths.
        n_paths = 100000

        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve.values

        # SDE constant.
#        rate, discount = \
#            self.sde_constant.paths(0, n_paths, seed=0, antithetic=True)
#        discount = self.sde_constant.discount_adjustment(discount,
#            self.sde_constant.discount_curve_eg)

        self.sde_constant.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_constant.discount_adjustment(
            self.sde_constant.discount_paths,
            self.sde_constant.discount_curve_eg)

        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))
        self.assertTrue(np.max(diff) < 2.59e-3)

        # SDE piecewise.
#        rate, discount = \
#            self.sde_piecewise1.paths(0, n_paths, seed=0, antithetic=True)
#        discount = self.sde_piecewise1.discount_adjustment(discount,
#            self.sde_piecewise1.discount_curve_eg)

        self.sde_piecewise1.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_piecewise1.discount_adjustment(
            self.sde_piecewise1.discount_paths,
            self.sde_piecewise1.discount_curve_eg)

        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))
        self.assertTrue(np.max(diff) < 2.59e-3)

        # SDE general.
#        rate, discount = \
#            self.sde_general1.paths(0, n_paths, seed=0, antithetic=True)
#        discount = self.sde_general1.discount_adjustment(discount,
#            self.sde_general1.discount_curve_eg)

        self.sde_general1.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_general1.discount_adjustment(
            self.sde_general1.discount_paths,
            self.sde_general1.discount_curve_eg)

        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))
        self.assertTrue(np.max(diff) < 2.50e-3)

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
        n_paths = 100000

        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve.values

        # SDE piecewise.
#        rate, discount = \
#            self.sde_piecewise2.paths(0, n_paths, seed=0, antithetic=True)
#        discount = self.sde_piecewise2.discount_adjustment(discount,
#            self.sde_piecewise2.discount_curve_eg)

        self.sde_piecewise2.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_piecewise2.discount_adjustment(
            self.sde_piecewise2.discount_paths,
            self.sde_piecewise2.discount_curve_eg)

        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))
        self.assertTrue(np.max(diff) < 9.1e-3)

        # SDE general.
#        rate, discount = \
#            self.sde_general2.paths(0, n_paths, seed=0, antithetic=True)
#        discount = self.sde_general2.discount_adjustment(discount,
#            self.sde_general2.discount_curve_eg)

        self.sde_general2.paths(0, n_paths, seed=0, antithetic=True)
        discount = self.sde_general2.discount_adjustment(
            self.sde_general2.discount_paths,
            self.sde_general2.discount_curve_eg)

        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))
        self.assertTrue(np.max(diff) < 8.86e-3)

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

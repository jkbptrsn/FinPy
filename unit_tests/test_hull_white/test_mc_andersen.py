import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import mc_andersen as sde
from models.hull_white import sde as sde_old
from models.hull_white import misc as misc_hw
from unit_tests.test_hull_white import input
from utils import misc

plot_results = True
print_results = False


class Misc(unittest.TestCase):

    def setUp(self) -> None:
        # Event dates in year fractions.
        self.event_grid = np.arange(31)
        # Speed of mean reversion.
        self.kappa_scalar = 0.015
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = \
            misc.DiscreteFunc("kappa1", self.event_grid, self.kappa_vector1)
        # Volatility.
        self.vol_scalar = 0.005
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        self.vol1 = \
            misc.DiscreteFunc("vol1", self.event_grid, self.vol_vector1)
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            if idx % 2 == 0:
                self.vol_vector2[idx] = self.vol_vector1[idx]
            else:
                self.vol_vector2[idx] = 2 * self.vol_vector1[idx]
        self.vol2 = \
            misc.DiscreteFunc("vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = \
            misc.DiscreteFunc("discount", self.event_grid,
                              np.ones(self.event_grid.size))
        self.sde1 = sde.SDEGeneral(self.kappa1, self.vol1,
                                   self.discount_curve, self.event_grid,
                                   int_step_size=1 / 12)
        self.sde2 = sde.SDEGeneral(self.kappa1, self.vol2,
                                   self.discount_curve, self.event_grid,
                                   int_step_size=1 / 360)

    def test_y_constant(self):
        """Test numerical evaluation of the y-function."""
        y_constant = misc_hw.y_constant(self.kappa_scalar, self.vol_scalar,
                                        self.event_grid)
        y_piecewise = misc_hw.y_piecewise(self.kappa_scalar, self.vol_vector1,
                                          self.event_grid)
        y_general, _ = misc_hw.y_general(self.sde1.int_grid,
                                         self.sde1.int_event_idx,
                                         self.sde1.int_kappa_step,
                                         self.sde1.vol_ig,
                                         self.sde1.event_grid)
        for idx, (y1, y2, y3) in \
                enumerate(zip(y_constant, y_piecewise, y_general)):
            if idx > 0:
                diff1 = abs(y1 - y2) / y1
                diff2 = abs(y1 - y3) / y1
                if print_results:
                    print(diff1, diff2)
                self.assertTrue(diff1 < 1.0e-15)
                self.assertTrue(diff2 < 1.0e-6)

    def test_y_piecewise(self):
        """Test numerical evaluation of the y-function.

        TODO: Check y_general calculation. The approximation seems too crude...
        """
        y_piecewise = misc_hw.y_piecewise(self.kappa_scalar, self.vol_vector2,
                                          self.event_grid)
        y_general, _ = misc_hw.y_general(self.sde2.int_grid,
                                         self.sde2.int_event_idx,
                                         self.sde2.int_kappa_step,
                                         self.sde2.vol_ig,
                                         self.sde2.event_grid)
        for idx, (y1, y2) in enumerate(zip(y_piecewise, y_general)):
            if idx > 1:
                diff = abs(y1 - y2) / y1
                if print_results:
                    print(y1, y2, diff)
                self.assertTrue(diff < 1.0e-3)

    # TODO: Add tests for int_y and double_int_y functions...


class SDE(unittest.TestCase):

    # TODO: Compare SDE classes like below. For time-independent speed
    #  of mean reversion and vol-strip, SDEConstant and SDEPiecewise
    #  should produce the same scenarios (within machine precision)
    #  when random number generator is reset between initializations...
    #  SDEGeneral should come close...
    # def test_sde_classes(self):
    #     """Test the classes SDEConstant and SDE.
    #
    #     In the case of both constant speed of mean reversion and
    #     constant volatility, the time-dependent mean and variance of
    #     the pseudo short rate and discount processes, respectively,
    #     should be.
    #     """
    #     # Event dates in year fractions.
    #     event_grid = np.arange(11)
    #     # Speed of mean reversion.
    #     kappa_const = 0.03
    #     # Volatility.
    #     vol_const = 0.02
    #     # Speed of mean reversion strip.
    #     kappa = np.array([np.arange(2), kappa_const * np.ones(2)])
    #     kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
    #     # Volatility strip.
    #     vol = np.array([np.arange(2), vol_const * np.ones(2)])
    #     vol = misc.DiscreteFunc("vol", vol[0], vol[1])
    #     # Number of Monte-Carlo paths.
    #     n_paths = 100000
    #     # SDE objects.
    #     hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
    #     hw_const = \
    #         sde.SDEConstant(kappa, vol, event_grid, int_step_size=1 / 52)
    #     # Pseudo rate and discount factors.
    #     rate, discount = hw.paths(0, n_paths, seed=0)
    #     rate_const, discount_const = hw_const.paths(0, n_paths, seed=0)
    #     # Compare trajectories.
    #     diff_rate = np.abs(rate[1:, :] - rate_const[1:, :]) / rate_const[1:, :]
    #     diff_rate = np.max(diff_rate)
    #     diff_discount = \
    #         np.abs(discount[1:, :] - discount_const[1:, :]) \
    #         / discount_const[1:, :]
    #     diff_discount = np.max(diff_discount)
    #     print(diff_rate, diff_discount)
    #     self.assertTrue(diff_rate < 8.3e-3)
    #     self.assertTrue(diff_discount < 1.7e-4)
    #     # Compare mean and variance of pseudo short rate and discount
    #     # processes, respectively.
    #     diff_rate_mean = hw.rate_mean[1:, :] - hw_const.rate_mean[1:, :]
    #     diff_rate_mean = np.abs(diff_rate_mean) / hw_const.rate_mean[1:, :]
    #     # print(np.max(diff_rate_mean[:, 0]), np.max(diff_rate_mean[:, 1]))
    #     self.assertTrue(np.max(diff_rate_mean[:, 0]) < 1.0e-10)
    #     self.assertTrue(np.max(diff_rate_mean[:, 1]) < 1.4e-7)
    #     diff_rate_var = hw.rate_variance[1:] - hw_const.rate_variance[1:]
    #     diff_rate_var = np.abs(diff_rate_var / hw_const.rate_variance[1:])
    #     # print(np.max(diff_rate_var))
    #     self.assertTrue(np.max(diff_rate_var) < 1.2e-7)
    #     diff_discount_mean = \
    #         hw.discount_mean[1:, :] - hw_const.discount_mean[1:, :]
    #     diff_discount_mean = \
    #         np.abs(diff_discount_mean) / hw_const.discount_mean[1:, :]
    #     # print(np.max(diff_discount_mean[:, 0]), np.max(diff_discount_mean[:, 1]))
    #     self.assertTrue(np.max(diff_discount_mean[:, 0]) < 2.8e-8)
    #     self.assertTrue(np.max(diff_discount_mean[:, 0]) < 1.9e-4)
    #     diff_discount_var = \
    #         hw.discount_variance[1:] - hw_const.discount_variance[1:]
    #     diff_discount_var = \
    #         np.abs(diff_discount_var / hw_const.discount_variance[1:])
    #     # print(np.max(diff_discount_var))
    #     self.assertTrue(np.max(diff_discount_var) < 1.9e-4)
    #     diff_cov = hw.covariance[1:] - hw_const.covariance[1:]
    #     diff_cov = np.abs(diff_cov / hw_const.covariance[1:])
    #     # print(np.max(diff_cov))
    #     self.assertTrue(np.max(diff_cov) < 5.7e-4)

    def test_zero_coupon_bond_pricing(self):
        """Test Monte-Carlo evaluation of zero-coupon bond price.

        Compare closed-form expression and Monte-Carlo simulation of
        zero-coupon bonds with different maturities.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip.
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([2, 2, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
#        vol = np.array([np.arange(10),
#                        vol_const * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = np.array([np.arange(10),
                        vol_const * np.ones(10)])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 1000

        # SDE object.
        hw_constant = sde.SDEConstant(kappa, vol, discount_curve, event_grid)
        rate, discount = hw_constant.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde_old.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        print(diff, np.max(diff))

        # SDE object.
        hw_piecewise = sde.SDEPiecewise(kappa, vol, discount_curve, event_grid)
        rate, discount = \
            hw_piecewise.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde_old.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))

        # SDE object.
        hw_general = sde.SDEGeneral(kappa, vol, discount_curve, event_grid)
        rate, discount = \
            hw_general.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde_old.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))

        if plot_results:
            plt.plot(event_grid, hw_constant.y_eg, "-b")
            plt.plot(event_grid, hw_piecewise.y_eg, "-r")
            plt.plot(event_grid, hw_general.y_eg, "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.rate_mean[:, 0], "-b")
            plt.plot(event_grid, hw_piecewise.rate_mean[:, 0], "-r")
            plt.plot(event_grid, hw_general.rate_mean[:, 0], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.rate_mean[:, 1], "-b")
            plt.plot(event_grid, hw_piecewise.rate_mean[:, 1], "-r")
            plt.plot(event_grid, hw_general.rate_mean[:, 1], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.discount_mean[:, 0], "-b")
            plt.plot(event_grid, hw_piecewise.discount_mean[:, 0], "-r")
            plt.plot(event_grid, hw_general.discount_mean[:, 0], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.discount_mean[:, 1], "-b")
            plt.plot(event_grid, hw_piecewise.discount_mean[:, 1], "-r")
            plt.plot(event_grid, hw_general.discount_mean[:, 1], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.covariance, "-b")
            plt.plot(event_grid, hw_piecewise.covariance, "-r")
            plt.plot(event_grid, hw_general.covariance, "-k")
            plt.show()

        #self.assertTrue(np.max(diff) < 2.4e-5)

    def test_zero_coupon_bond_pricing_2(self):
        """Test Monte-Carlo evaluation of zero-coupon bond price.

        Compare closed-form expression and Monte-Carlo simulation of
        zero-coupon bonds with different maturities.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip.
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([2, 2, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 1000

        # SDE object.
        hw_constant = sde.SDEConstant(kappa, vol, discount_curve, event_grid)
        rate, discount = hw_constant.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde_old.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))

        # SDE object.
        hw_piecewise = sde.SDEPiecewise(kappa, vol, discount_curve, event_grid)
        rate, discount = \
            hw_piecewise.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde_old.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))

        # SDE object.
        hw_general = sde.SDEGeneral(kappa, vol, discount_curve, event_grid)
        rate, discount = \
            hw_general.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde_old.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        if print_results:
            print(diff, np.max(diff))

        if plot_results:
            plt.plot(event_grid, hw_constant.y_eg, "-b")
            plt.plot(event_grid, hw_piecewise.y_eg, "-r")
            plt.plot(event_grid, hw_general.y_eg, "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.rate_mean[:, 0], "-b")
            plt.plot(event_grid, hw_piecewise.rate_mean[:, 0], "-r")
            plt.plot(event_grid, hw_general.rate_mean[:, 0], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.rate_mean[:, 1], "-b")
            plt.plot(event_grid, hw_piecewise.rate_mean[:, 1], "-r")
            plt.plot(event_grid, hw_general.rate_mean[:, 1], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.discount_mean[:, 0], "-b")
            plt.plot(event_grid, hw_piecewise.discount_mean[:, 0], "-r")
            plt.plot(event_grid, hw_general.discount_mean[:, 0], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.discount_mean[:, 1], "-b")
            plt.plot(event_grid, hw_piecewise.discount_mean[:, 1], "-r")
            plt.plot(event_grid, hw_general.discount_mean[:, 1], "-k")
            plt.show()

            plt.plot(event_grid, hw_constant.covariance, "-b")
            plt.plot(event_grid, hw_piecewise.covariance, "-r")
            plt.plot(event_grid, hw_general.covariance, "-k")
            plt.show()


if __name__ == '__main__':
    unittest.main()

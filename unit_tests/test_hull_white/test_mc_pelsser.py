import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import mc_pelsser as sde
from models.hull_white import misc as misc_hw
from utils import data_types

plot_results = False
print_results = False


class Misc(unittest.TestCase):
    """Various functions used in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Event dates as year fractions from as-of date.
        self.event_grid = np.arange(16)
        # Speed of mean reversion strip.
        self.kappa_scalar = 0.02
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = data_types.DiscreteFunc(
            "kappa1", self.event_grid, self.kappa_vector1)
        # Constant vol strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        self.vol1 = data_types.DiscreteFunc(
            "vol1", self.event_grid, self.vol_vector1)
        # Piecewise-constant vol strip.
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            self.vol_vector2[idx] = (idx % 4 + 1) * self.vol_vector1[idx]
        self.vol2 = data_types.DiscreteFunc(
            "vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = data_types.DiscreteFunc(
            "discount", self.event_grid, np.ones(self.event_grid.size))
        # SDE object, constant vol strip.
        self.sde1 = sde.SdeExactGeneral(
            self.kappa1, self.vol1, self.discount_curve, self.event_grid,
            int_dt=1 / 100)
        # SDE object, piecewise-constant vol strip.
        self.sde2 = sde.SdeExactGeneral(
            self.kappa1, self.vol2, self.discount_curve, self.event_grid,
            int_dt=1 / 100)

    def test_alpha_constant(self):
        """Alpha-function with constant vol."""
        alpha_constant = misc_hw.alpha_constant(
            self.kappa_scalar, self.vol_scalar, self.event_grid)
        alpha_piecewise = misc_hw.alpha_piecewise(
            self.kappa_scalar, self.vol_vector1, self.event_grid)
        alpha_general = misc_hw.alpha_general(
            self.sde1.int_grid, self.sde1.int_event_idx,
            self.sde1.int_kappa_step_ig, self.sde1.vol_ig,
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
        alpha_piecewise = misc_hw.alpha_piecewise(
            self.kappa_scalar, self.vol_vector2, self.event_grid)
        alpha_general = misc_hw.alpha_general(
            self.sde2.int_grid, self.sde2.int_event_idx,
            self.sde2.int_kappa_step_ig, self.sde2.vol_ig,
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
        self.assertTrue(np.max(diff) < 4.4e-3)

    def test_int_alpha_constant(self):
        """Integral of alpha-function with constant vol."""
        int_alpha_constant = misc_hw.int_alpha_constant(
            self.kappa_scalar, self.vol_scalar, self.event_grid)
        int_alpha_piecewise = misc_hw.int_alpha_piecewise(
            self.kappa_scalar, self.vol_vector1, self.event_grid)
        int_alpha_general = misc_hw.int_alpha_general(
            self.sde1.int_grid, self.sde1.int_event_idx,
            self.sde1.int_kappa_step_ig, self.sde1.vol_ig,
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
        int_alpha_piecewise = misc_hw.int_alpha_piecewise(
            self.kappa_scalar, self.vol_vector2, self.event_grid)
        int_alpha_general = misc_hw.int_alpha_general(
            self.sde2.int_grid, self.sde2.int_event_idx,
            self.sde2.int_kappa_step_ig, self.sde2.vol_ig,
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
        self.assertTrue(np.max(diff) < 4.2e-3)


class SDE(unittest.TestCase):
    """SDE classes in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Event dates as year fractions from as-of date.
        self.event_grid = np.arange(16)
        self.event_grid_mc = np.arange(151) / 10
        # Speed of mean reversion strip.
        self.kappa_scalar = 0.15
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = data_types.DiscreteFunc(
            "kappa1", self.event_grid, self.kappa_vector1)
        # Constant vol strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        self.vol1 = data_types.DiscreteFunc(
            "vol1", self.event_grid, self.vol_vector1)
        # Piecewise-constant vol strip.
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            if idx % 2 == 0:
                self.vol_vector2[idx] = self.vol_vector1[idx]
            else:
                self.vol_vector2[idx] = 2 * self.vol_vector1[idx]
        self.vol2 = data_types.DiscreteFunc(
            "vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = data_types.DiscreteFunc(
            "discount", self.event_grid, np.ones(self.event_grid.size))
        self.discount_curve_mc = data_types.DiscreteFunc(
            "discount", self.event_grid_mc, np.ones(self.event_grid_mc.size))
        # SDE objects.
        self.sde_constant = sde.SdeExactConstant(
            self.kappa1, self.vol1, self.discount_curve, self.event_grid)
        self.sde_piecewise1 = sde.SdeExactPiecewise(
            self.kappa1, self.vol1, self.discount_curve, self.event_grid)
        self.sde_piecewise2 = sde.SdeExactPiecewise(
            self.kappa1, self.vol2, self.discount_curve, self.event_grid)
        self.sde_general1 = sde.SdeExactGeneral(
            self.kappa1, self.vol1, self.discount_curve, self.event_grid,
            int_dt=1 / 50)
        self.sde_general2 = sde.SdeExactGeneral(
            self.kappa1, self.vol2, self.discount_curve, self.event_grid,
            int_dt=1 / 50)
        self.sde_euler1 = sde.SdeEuler(
            self.kappa1, self.vol1, self.discount_curve_mc, self.event_grid_mc)
        self.sde_euler2 = sde.SdeEuler(
            self.kappa1, self.vol2, self.discount_curve_mc, self.event_grid_mc)
        # Integral of alpha-function.
        self.int_alpha1 = misc_hw.int_alpha_constant(
            self.kappa_scalar, self.vol_scalar, self.event_grid)
        self.int_alpha1_mc = misc_hw.int_alpha_constant(
            self.kappa_scalar, self.vol_scalar, self.event_grid_mc)
        vol_eg = self.vol2.interpolation(self.event_grid)
        self.int_alpha2 = misc_hw.int_alpha_piecewise(
            self.kappa_scalar, vol_eg, self.event_grid)
        vol_eg = self.vol2.interpolation(self.event_grid_mc)
        self.int_alpha2_mc = misc_hw.int_alpha_piecewise(
            self.kappa_scalar, vol_eg, self.event_grid_mc)

    def test_sde_constant_vol(self):
        """Test SDE classes for constant vol-strip."""
        # Number of Monte-Carlo paths.
        n_paths = 100000
        seed = 0

        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve.values

        # SDE constant.
        self.sde_constant.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha1))
        discount = self.sde_constant.discount_adjustment(
            self.sde_constant.discount_paths,
            self.sde_constant.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE constant: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 2.6e-3)

        # SDE piecewise.
        self.sde_piecewise1.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha1))
        discount = self.sde_piecewise1.discount_adjustment(
            self.sde_piecewise1.discount_paths,
            self.sde_piecewise1.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE piecewise: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 2.6e-3)

        # SDE general.
        self.sde_general1.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha1))
        discount = self.sde_general1.discount_adjustment(
            self.sde_general1.discount_paths,
            self.sde_general1.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE general: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 2.6e-3)

        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve_mc.values

        # SDE Euler.
        self.sde_euler1.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha1_mc))
        discount = self.sde_euler1.discount_adjustment(
            self.sde_euler1.discount_paths,
            self.sde_euler1.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE Euler: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 8.2e-4)

    def test_sde_piecewise_vol(self):
        """Test SDE classes for piecewise-constant vol-strip."""
        # Number of Monte-Carlo paths.
        n_paths = 100000
        seed = 0

        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve.values

        # SDE piecewise.
        self.sde_piecewise2.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha2))
        discount = self.sde_piecewise2.discount_adjustment(
            self.sde_piecewise2.discount_paths,
            self.sde_piecewise2.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE piecewise: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 9.1e-3)

        # SDE general.
        self.sde_general2.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha2))
        discount = self.sde_general2.discount_adjustment(
            self.sde_general2.discount_paths,
            self.sde_general2.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE general: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 9.8e-3)

        # Zero-coupon bond price at all events. Analytical results.
        price_a = self.discount_curve_mc.values

        # SDE Euler.
        self.sde_euler2.paths(0, n_paths, seed=seed, antithetic=True)
        adjustment = np.cumprod(np.exp(-self.int_alpha2_mc))
        discount = self.sde_euler2.discount_adjustment(
            self.sde_euler2.discount_paths,
            self.sde_euler2.discount_curve_eg * adjustment)
        # Zero-coupon bond price at all events. Monte-Carlo estimates.
        price_n = np.mean(discount, axis=1)
        # Maximum relative difference.
        diff = np.abs((price_n[1:] - price_a[1:]) / price_a[1:])
        if print_results:
            print(f"SDE Euler: Diff = {np.max(diff)}")
        self.assertTrue(np.max(diff) < 1.9e-2)

import unittest

import matplotlib.pyplot as plt
import numpy as np

from unit_tests.test_hull_white import input
from models.hull_white import european_option
from models.hull_white import caplet_floorlet
from models.hull_white import put_option
from models.hull_white import sde
from models.hull_white import swap
from models.hull_white import swaption
from models.hull_white import zero_coupon_bond
from utils import misc


class SDE(unittest.TestCase):

    def test_y_function(self):
        """Test numerical evaluation of the y-function.

        In the case of both constant speed of mean reversion and
        constant volatility, the y-function has a closed form. See
        proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        # Event dates in year fractions.
        event_grid = np.arange(0, 31, 2)
        # Speed of mean reversion.
        kappa_const = 0.05
        two_kappa = 2 * kappa_const
        # Volatility.
        vol_const = 0.02
        # Closed-form evaluation of y-function.
        y_analytical = \
            vol_const ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa
        # Speed of mean reversion strip.
        kappa = np.array([np.arange(2), kappa_const * np.ones(2)])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(2), vol_const * np.ones(2)])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        for idx, y_numerical in enumerate(hw.y_event_grid):
            if idx >= 1:
                diff = abs(y_analytical[idx] - y_numerical) / y_analytical[idx]
                # print(idx, y_analytical[idx], diff)
                self.assertTrue(diff < 3.1e-7)

    def test_sde_classes(self):
        """Test the classes SDEConstant and SDE.

        In the case of both constant speed of mean reversion and
        constant volatility, the time-dependent mean and variance of
        the pseudo short rate and discount processes, respectively,
        should be.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.03
        # Volatility.
        vol_const = 0.02
        # Speed of mean reversion strip.
        kappa = np.array([np.arange(2), kappa_const * np.ones(2)])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(2), vol_const * np.ones(2)])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Number of Monte-Carlo paths.
        n_paths = 100000
        # SDE objects.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        hw_const = \
            sde.SDEConstant(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors.
        rate, discount = hw.paths(0, n_paths, seed=0)
        rate_const, discount_const = hw_const.paths(0, n_paths, seed=0)
        # Compare trajectories.
        diff_rate = np.abs(rate[1:, :] - rate_const[1:, :]) / rate_const[1:, :]
        diff_rate = np.max(diff_rate)
        diff_discount = \
            np.abs(discount[1:, :] - discount_const[1:, :]) \
            / discount_const[1:, :]
        diff_discount = np.max(diff_discount)
        print(diff_rate, diff_discount)
        self.assertTrue(diff_rate < 8.3e-3)
        self.assertTrue(diff_discount < 1.7e-4)
        # Compare mean and variance of pseudo short rate and discount
        # processes, respectively.
        diff_rate_mean = hw.rate_mean[1:, :] - hw_const.rate_mean[1:, :]
        diff_rate_mean = np.abs(diff_rate_mean) / hw_const.rate_mean[1:, :]
        # print(np.max(diff_rate_mean[:, 0]), np.max(diff_rate_mean[:, 1]))
        self.assertTrue(np.max(diff_rate_mean[:, 0]) < 1.0e-10)
        self.assertTrue(np.max(diff_rate_mean[:, 1]) < 1.4e-7)
        diff_rate_var = hw.rate_variance[1:] - hw_const.rate_variance[1:]
        diff_rate_var = np.abs(diff_rate_var / hw_const.rate_variance[1:])
        # print(np.max(diff_rate_var))
        self.assertTrue(np.max(diff_rate_var) < 1.2e-7)
        diff_discount_mean = \
            hw.discount_mean[1:, :] - hw_const.discount_mean[1:, :]
        diff_discount_mean = \
            np.abs(diff_discount_mean) / hw_const.discount_mean[1:, :]
        # print(np.max(diff_discount_mean[:, 0]), np.max(diff_discount_mean[:, 1]))
        self.assertTrue(np.max(diff_discount_mean[:, 0]) < 2.8e-8)
        self.assertTrue(np.max(diff_discount_mean[:, 0]) < 1.9e-4)
        diff_discount_var = \
            hw.discount_variance[1:] - hw_const.discount_variance[1:]
        diff_discount_var = \
            np.abs(diff_discount_var / hw_const.discount_variance[1:])
        # print(np.max(diff_discount_var))
        self.assertTrue(np.max(diff_discount_var) < 1.9e-4)
        diff_cov = hw.covariance[1:] - hw_const.covariance[1:]
        diff_cov = np.abs(diff_cov / hw_const.covariance[1:])
        # print(np.max(diff_cov))
        self.assertTrue(np.max(diff_cov) < 5.7e-4)

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
            np.array([np.array([2, 3, 7]), kappa_const * np.array([2, 1, 2])])
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
        n_paths = 100000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        rate, discount = hw.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde.discount_adjustment(discount, discount_curve)
        # Analytical results.
        price_a = discount_curve.values
        # Monte-Carlo estimates.
        price_n = discount.sum(axis=1) / n_paths
        # Maximum relative difference.
        diff = np.abs((price_n - price_a) / price_a)
        print(diff, np.max(diff))
        self.assertTrue(np.max(diff) < 2.4e-5)

    def test_coupon_bond_pricing(self):
        """Test Monte-Carlo evaluation of coupon bond price.

        Compare closed-form expression and Monte-Carlo simulation of 10Y
        coupon bond with yearly coupon frequency.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Yearly coupon.
        coupon = 0.03
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip.
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([2, 1, 2])])
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
        n_paths = 100000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        rate, discount = hw.paths(0, n_paths, seed=0, antithetic=True)
        discount = sde.discount_adjustment(discount, discount_curve)
        # Analytical discount factors.
        discount_a = discount_curve.values
        # Monte-Carlo discount factors.
        discount_n = discount.sum(axis=1) / n_paths
        # Prices contributions from coupon payments and principal.
        price_a_c = np.sum(coupon * discount_a[1:])
        price_a_p = discount_a[-1]
        price_n_c = np.sum(coupon * discount_n[1:])
        price_n_p = discount_n[-1]
        # Relative differences.
        diff_c = abs((price_n_c - price_a_c) / price_a_c)
        diff_p = abs((price_n_p - price_a_p) / price_a_p)
        # print(price_a_c, diff_c)
        # print(price_a_p, diff_p)
        self.assertTrue(diff_c < 2.0e-6)
        self.assertTrue(diff_p < 1.6e-5)

    def test_call_option_pricing_1(self):
        """Test Monte-Carlo evaluation of call option price.

        Compare closed-form expression and Monte-Carlo simulation of
        European call option written on zero coupon bond. Also, a
        comparison of SDE classes, hence, kappa and vol are constant.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip -- constant kappa!
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.01 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 100000
        # SDE objects.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        hw_const = \
            sde.SDEConstant(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors.
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        r_pseudo_const, d_pseudo_const = \
            hw_const.paths(0, n_paths, seed=0, antithetic=True)
        # Expiry index.
        expiry_idx = 5
        # Maturity index.
        maturity_idx = event_grid.size - 1
        # Strike value.
        strike = 0.65
        # Call option object.
        call = call_option.EuropeanOption(kappa, vol, discount_curve, event_grid,
                                          strike, expiry_idx, maturity_idx,
                                          int_step_size=1 / 52)
        # Zero-coupon bond object.
        bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                       maturity_idx, int_step_size=1 / 52)
        # Threshold.
        threshold = np.array([5e-5, 5e-5, 5e-5, 7e-5, 2e-4, 5e-3])
        for s in range(2, 13, 2):
            # New discount curve on event_grid.
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curves.
            call.discount_curve = discount_curve
            call.zcbond.discount_curve = discount_curve
            bond.discount_curve = discount_curve
            # Call option price, analytical.
            call_price_a = call.price(0, 0)
            # Call option price, numerical, SDE class.
            # Zero-coupon bond price at expiry.
            bond_price = bond.price(r_pseudo[expiry_idx], expiry_idx)
            # Call option payoff.
            payoff = np.maximum(bond_price - strike, 0)
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Monte-Carlo estimate of call option price at time zero.
            call_price_n = np.sum(discount[expiry_idx] * payoff) / n_paths
            # Relative difference.
            diff = abs((call_price_a - call_price_n) / call_price_a)
            # Call option price, numerical, SDEConstant class.
            # Zero-coupon bond price at expiry.
            bond_price = bond.price(r_pseudo_const[expiry_idx], expiry_idx)
            # Call option payoff.
            payoff = np.maximum(bond_price - strike, 0)
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo_const, discount_curve)
            # Monte-Carlo estimate of call option price at time zero.
            call_price_n = np.sum(discount[expiry_idx] * payoff) / n_paths
            # Relative difference.
            diff_const = abs((call_price_a - call_price_n) / call_price_a)
            # print(s, call_price_a, diff, diff_const)
            self.assertTrue(diff < threshold[s // 2 - 1])
            self.assertTrue(diff_const < threshold[s // 2 - 1])

    def test_call_option_pricing_2(self):
        """Test Monte-Carlo evaluation of call option price.

        Compare closed-form expression and Monte-Carlo simulation of
        European call option written on zero coupon bond.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip -- constant kappa!
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([3, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 100000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        # Expiry index.
        expiry_idx = 5
        # Maturity index.
        maturity_idx = event_grid.size - 1
        # Strike value.
        strike = 0.65
        # Call option object.
        call = call_option.EuropeanOption(kappa, vol, discount_curve, event_grid, strike,
                                          expiry_idx, maturity_idx, int_step_size=1 / 52)
        # Zero-coupon bond object.
        bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                       maturity_idx, int_step_size=1 / 52)
        # Threshold
        threshold = np.array([6e-5, 4e-5, 2e-4, 3e-4, 1e-3, 7e-3])
        for s in range(2, 13, 2):
            # New discount curve on event_grid.
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curves.
            call.discount_curve = discount_curve
            call.zcbond.discount_curve = discount_curve
            bond.discount_curve = discount_curve
            # Call option price, analytical.
            call_price_a = call.price(0, 0)
            # Call option price, numerical.
            # Zero-coupon bond price at expiry.
            bond_price = bond.price(r_pseudo[expiry_idx], expiry_idx)
            # Call option payoff.
            payoff = np.maximum(bond_price - strike, 0)
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Monte-Carlo estimate of call option price at time zero.
            call_price_n = np.sum(discount[expiry_idx] * payoff) / n_paths
            # Relative difference.
            diff = abs((call_price_a - call_price_n) / call_price_a)
            # print(s, call_price_a, call_price_n, diff)
            self.assertTrue(diff < threshold[s // 2 - 1])

    def test_put_option_pricing(self):
        """Test Monte-Carlo evaluation of put option price.

        Compare closed-form expression and Monte-Carlo simulation of
        European put option written on zero coupon bond.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip -- constant kappa!
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([3, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 100000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        # Expiry index.
        expiry_idx = 5
        # Maturity index.
        maturity_idx = event_grid.size - 1
        # Strike value.
        strike = 0.8
        # Put option object.
        put = put_option.Put(kappa, vol, discount_curve, event_grid, strike,
                             expiry_idx, maturity_idx, int_step_size=1 / 52)
        # Zero-coupon bond object.
        bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                       maturity_idx, int_step_size=1 / 52)
        # Threshold
        threshold = np.array([6e-3, 2e-3, 3e-4, 5e-4, 6e-5, 3e-6])
        for s in range(2, 13, 2):
            # New discount curve on event_grid.
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curves.
            put.discount_curve = discount_curve
            put.zcbond.discount_curve = discount_curve
            bond.discount_curve = discount_curve
            # Put option price, analytical.
            put_price_a = put.price(0, 0)
            # Put option price, numerical.
            # Zero-coupon bond price at expiry.
            bond_price = bond.price(r_pseudo[expiry_idx], expiry_idx)
            # Put option payoff.
            payoff = np.maximum(strike - bond_price, 0)
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Monte-Carlo estimate of put option price at time zero.
            put_price_n = np.sum(discount[expiry_idx] * payoff) / n_paths
            # Relative difference.
            diff = abs((put_price_a - put_price_n) / put_price_a)
            # print(s, put_price_a, put_price_n, diff)
            self.assertTrue(diff < threshold[s // 2 - 1])

    def test_swap_pricing(self):
        """Test Monte-Carlo evaluation of swap price.

        Compare closed-form expression and Monte-Carlo simulation of
        fixed-for-floating swap.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip.
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([2, 1, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([3, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 100000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        # Fixed rate of swap
        fixed_rate = 0.03
        # Swap object
        swap_obj = swap.Swap(kappa, vol, discount_curve, event_grid,
                             fixed_rate, int_step_size=1 / 52)
        # Threshold
        threshold = np.array([2e-4, 4e-4, 6e-4, 2e-4, 8e-5, 5e-5])
        for s in range(2, 13, 2):
            # New discount curve on event_grid.
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curve.
            swap_obj.zcbond.discount_curve = discount_curve
            # Swap price, analytical.
            swap_price_a = swap_obj.price(0, 0)
            # Swap price, numerical.
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Simple forward rate
            simple_forward_rate = np.zeros(discount.shape)
            for idx, tau in enumerate(np.diff(event_grid)):
                discount_forward = \
                    swap_obj.zcbond._calc_price(r_pseudo[idx], idx, idx + 1)
                simple_forward_rate[idx] = (1 / discount_forward - 1) / tau
            # Monte-Carlo estimate of swap price.
            swap_price_n = \
                discount[1:] * (simple_forward_rate[:-1] - fixed_rate)
            swap_price_n = swap_price_n.transpose() * np.diff(event_grid)
            swap_price_n = swap_price_n.transpose()
            swap_price_n = np.sum(swap_price_n) / n_paths
            # Relative difference.
            diff = abs((swap_price_a - swap_price_n) / swap_price_a)
            # print(s, swap_price_a, swap_price_n, diff)
            self.assertTrue(diff < threshold[s // 2 - 1])

    def test_caplet_pricing(self):
        """Test Monte-Carlo evaluation of caplet price.

        Compare closed-form expression and Monte-Carlo simulation of
        caplet.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip -- constant kappa!
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([3, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 100000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        # Fixed rate of caplet
        fixed_rate = 0.01
        # Expiry index.
        expiry_idx = 5
        # Maturity index.
        maturity_idx = event_grid.size - 1
        # Caplet object
        caplet_obj = \
            caplet_floorlet.Caplet(kappa, vol, discount_curve, event_grid,
                                   expiry_idx, maturity_idx, fixed_rate,
                                   int_step_size=1 / 52)
        # Threshold
        threshold = np.array([3e-4, 4e-4, 4e-5, 7e-5, 4e-5, 3e-5])
        for s in range(2, 13, 2):
            # New discount curve on event_grid.
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curve.
            caplet_obj.zcbond.discount_curve = discount_curve
            # Caplet price, analytical.
            caplet_price_a = caplet_obj.price(0, 0)
            # Caplet price, numerical.
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Simple forward rate from expiry to maturity
            discount_forward = \
                caplet_obj.zcbond._calc_price(r_pseudo[expiry_idx], expiry_idx,
                                              maturity_idx)
            tau = event_grid[maturity_idx] - event_grid[expiry_idx]
            simple_forward_rate = (1 / discount_forward - 1) / tau
            # Monte-Carlo estimate of caplet price.
            caplet_price_n = \
                tau * np.maximum(simple_forward_rate - fixed_rate, 0)
            # Payoff is payed at maturity, and not expiry...
            caplet_price_n *= discount[maturity_idx]
            caplet_price_n = np.sum(caplet_price_n) / n_paths
            # Relative difference.
            diff = abs((caplet_price_a - caplet_price_n) / caplet_price_a)
            # print(s, caplet_price_a, caplet_price_n, diff)
            self.assertTrue(diff < threshold[s // 2 - 1])

    def test_swaption_pricing(self):
        """Test Monte-Carlo evaluation of swaption price.

        Compare closed-form expression and Monte-Carlo simulation of
        swaption.
        """
        # Event dates in year fractions.
        event_grid = np.arange(11)
        # Speed of mean reversion.
        kappa_const = 0.015
        # Volatility.
        vol_const = 0.005
        # Speed of mean reversion strip -- constant kappa!
        kappa = \
            np.array([np.array([2, 3, 7]), kappa_const * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip.
        vol = np.array([np.arange(10),
                        vol_const * np.array([3, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve.
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid, discount_curve)
        # Number of Monte-Carlo paths.
        n_paths = 10000
        # SDE object.
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        # Fixed rate of swaption
        fixed_rate = 0.03
        # Expiry index.
        expiry_idx = 5
        # Maturity index.
        maturity_idx = event_grid.size - 1

        # swaption object
        swaption_obj = swaption.Payer(kappa, vol, discount_curve, event_grid,
                                      expiry_idx, maturity_idx, fixed_rate,
                                      int_step_size=1 / 52)

        # Threshold
        threshold = np.array([3e-4, 4e-4, 4e-5, 7e-5, 4e-5, 3e-5])

        for s in range(2, 13, 2):
            # New discount curve on event_grid.
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curve.
            swaption_obj.swap.zcbond.discount_curve = discount_curve
            swaption_obj.zcbond.discount_curve = discount_curve
            swaption_obj.put.zcbond.discount_curve = discount_curve

            # Swaption price, analytical.
#            swaption_price_a = swaption_obj.price(0, 0)
            swaption_price_a = 1

            # Swaption price, numerical.
            # Discount factor.
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Swaption payoff.
            payoff = swaption_obj.swap.price(r_pseudo[expiry_idx], expiry_idx)
            payoff = discount[expiry_idx] * np.maximum(payoff, 0)
            # Monte-Carlo estimate of swaption price.
            swaption_price_n = np.sum(payoff) / n_paths
            # Relative difference.
            diff = abs((swaption_price_a - swaption_price_n) / swaption_price_a)
            # print(s, swaption_price_a, swaption_price_n, diff)
            # self.assertTrue(diff < threshold[s // 2 - 1])

    def test_refi_coupon_calc(self):
        """Calculate re-finance coupon..."""
        # Speed of mean reversion strip
        kappa = np.array([np.array([0, 10]), 0.023 * np.array([1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 20]),
                        np.array([0.0165, 0.0143, 0.0140, 0.0132, 0.0128,
                                  0.0103, 0.0067, 0.0096, 0.0087, 0.0091,
                                  0.0098])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve
        time_grid = np.array([0.09, 0.26, 0.5, 1, 1.5, 2, 3, 4,
                              5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
        rate_grid = np.array([-0.0034, 0.0005, 0.0061, 0.0135, 0.0179,
                              0.0202, 0.0224, 0.0237, 0.0246, 0.0252,
                              0.0256, 0.0261, 0.0265, 0.0270, 0.0277,
                              0.0281, 0.0267, 0.0249, 0.0233])
        discount_curve = np.exp(-rate_grid * time_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", time_grid,
                              discount_curve, interp_scheme="quadratic")
        # Quarterly payment dates for a 30Y bond
        event_grid = np.arange(4 * 30 + 1) / 4
        # Discount curve on event_grid
        discount_curve = discount_curve.interpolation(event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="quadratic")
        # SDE object
        n_paths = 10000
        hw = sde.SDE(kappa, vol, event_grid)
        rate, discount = hw.paths(0, n_paths, seed=0)
        discount = sde.discount_adjustment(discount, discount_curve)
        # Zero-coupon bond object
        maturity_idx = event_grid.size - 1
        bond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)
        for n in [10, 50, 100]:
#            print("Event idx: ", n)
#            print("Pseudo rates: \n", rate[n])
            n_payments = event_grid.size - (n + 1)
#            print("# of payments: ", n_payments)
            # All "maturities" from t_{n + 1} to T inclusive
            maturity_indices = np.arange(n + 1, event_grid.size)
            # Discount curve for each pseudo rate scenario
            # REVISE PRICE_VECTOR METHOD IN BOND CLASS!!!
            discount_vector = bond.price_vector(rate[n], n, maturity_indices)
#            print("Discount curves, MC scenarios: \n", discount_vector)
            sum_discount = discount_vector.sum(axis=0)
#            print("Sum discount: \n", sum_discount)
            # Construct pseudo rate grid
            rate_max = np.max(rate[n])
            rate_min = np.min(rate[n])
            rate_interval = rate_max - rate_min
#            print("Rate max, min, interval: ", rate_max, rate_min, rate_interval)
            rate_n_grid = 5
            rate_grid = np.arange(rate_n_grid) / (rate_n_grid - 1)
            rate_grid = rate_interval * rate_grid + rate_min
#            print("Rate grid: \n", rate_grid)
            # Discount curve for each pseudo rate grid point
            # REVISE PRICE_VECTOR METHOD IN BOND CLASS!!!
            discount_grid = bond.price_vector(rate_grid, n, maturity_indices)
#            print("Discount curves, rate grid: \n", discount_grid)
            sum_discount_grid = discount_grid.sum(axis=0)
            # Coupons of refinance bond on pseudo rate grid
            coupon_grid = np.zeros(rate_n_grid)
            for m in range(rate_n_grid):
                coupon_grid[m] = \
                    misc.calc_refinancing_coupon(n_payments,
                                                 sum_discount_grid[m])
#            print("Coupon grid: \n", coupon_grid)
            coupon_grid = \
                misc.DiscreteFunc("Coupon grid", rate_grid,
                                  coupon_grid, interp_scheme="quadratic")
            coupon_grid_int = coupon_grid.interpolation(rate[n])
            # Coupons of refinance bond for MC scenarios
            coupon_scenarios = np.zeros(n_paths)
            for m in range(n_paths):
                coupon_scenarios[m] = \
                    misc.calc_refinancing_coupon(n_payments, sum_discount[m])
#            print("Are all close: ",
#                  np.allclose(coupon_scenarios, coupon_grid_int,
#                              rtol=1e-5, atol=1e-5))
            self.assertTrue(np.allclose(coupon_scenarios, coupon_grid_int,
                                        rtol=1e-5, atol=1e-5))


if __name__ == '__main__':

    # Speed of mean reversion strip.
    kappa = input.kappa_strip
    # Volatility strip.
    vol = input.vol_strip
    # Discount curve.
    discount_curve = input.disc_curve

    # # Plot zero-coupon bond price curve.
    # time_grid_plot = 0.1 * np.arange(301)
    # plt.plot(time_grid_plot, discount_curve.interpolation(time_grid_plot))
    # plt.xlabel("Time")
    # plt.ylabel("Zero coupon bond price")
    # plt.show()

    event_grid_plot = 0.1 * np.arange(251)
    n_paths = 20000  # 200000
    hw = sde.SDE(kappa, vol, event_grid_plot, int_step_size=1 / 52)
    rate, discount = hw.paths(0, n_paths, seed=0)

    rate *= 100

    forward_rate = input.forward_rate_ext
    forward_rate_plot = forward_rate.interpolation(event_grid_plot)
    forward_rate_plot *= 100
    rate_adj = (rate.transpose() + forward_rate_plot).transpose()

    rate_avg = np.sum(rate, axis=1) / n_paths
    rate_adj_avg = np.sum(rate_adj, axis=1) / n_paths

    d_curve = discount_curve.interpolation(event_grid_plot)
    discount_curve_plot = \
        misc.DiscreteFunc("discount plot", event_grid_plot, d_curve,
                          interp_scheme="cubic")
    discount_adj = sde.discount_adjustment(discount, discount_curve_plot)

    discount_avg = np.sum(discount, axis=1) / n_paths
    discount_adj_avg = np.sum(discount_adj, axis=1) / n_paths

    n_bins = 100
    # Rate grid
    r_min = -10
    r_max = 15
    hist = np.histogram(rate_adj[1, :], bins=n_bins,
                        range=(r_min - 1, r_max + 1), density=True)
    r_grid = (hist[1][1:] + hist[1][:-1]) / 2
    # Discount grid
    d_min = 0
    d_max = 2
    hist = np.histogram(discount_adj[1, :], bins=n_bins, range=(d_min, d_max),
                        density=True)
    d_grid = (hist[1][1:] + hist[1][:-1]) / 2

    # Density plots
    X, Y = np.meshgrid(event_grid_plot, r_grid)
    Z = np.zeros(X.shape)
    Z_adj = np.zeros(X.shape)

    XD, YD = np.meshgrid(event_grid_plot, d_grid)
    ZD = np.zeros(X.shape)
    ZD_adj = np.zeros(X.shape)

    for n in range(event_grid_plot.size):
        hist = np.histogram(rate[n, :], bins=n_bins,
                            range=(r_min - 1, r_max + 1), density=True)
        Z[:, n] = hist[0]
        hist_adj = np.histogram(rate_adj[n, :], bins=n_bins,
                                range=(r_min - 1, r_max + 1), density=True)
        Z_adj[:, n] = hist_adj[0]

        hist = np.histogram(discount[n, :], bins=n_bins,
                            range=(d_min, d_max), density=True)
        ZD[:, n] = hist[0]
        hist = np.histogram(discount_adj[n, :], bins=n_bins,
                            range=(d_min, d_max), density=True)
        ZD_adj[:, n] = hist[0]

    # Density plot of pseudo short rate
    plt.pcolormesh(X, Y, Z, cmap=plt.colormaps['hot'], shading='gouraud')
    plt.xlabel("t [years]")
    plt.ylabel("Pseudo short rate x(t) = r(t) - f(0,t) [%]")
    clb = plt.colorbar(ticks=0.01 * np.arange(0, 21, 2))
    clb.set_label("Probability density")
    plt.ylim(-8, 13)
    plt.clim(0, 0.16)
    lgd1 = plt.plot(event_grid_plot, rate_avg, "-b")
    plt.legend(lgd1, ["E[x(t)]"])
    plt.show()

    # Density plot of pseudo discount curve
    plt.pcolormesh(XD, YD, ZD, cmap=plt.colormaps['hot'], shading='gouraud')
    plt.xlabel("t [years]")
    plt.ylabel("Pseudo discount curve")
    clb = plt.colorbar(ticks=0.1 * np.arange(0, 31, 5))
    clb.set_label("Probability density")
    plt.ylim(0.1, 1.9)
    plt.clim(0, 3)
    lgd1 = plt.plot(event_grid_plot, discount_avg, "-b")
    plt.legend(lgd1, ["E[D(t)]"])
    plt.show()

    # Density plot of short rate
    plt.pcolormesh(X, Y, Z_adj, cmap=plt.colormaps['hot'], shading='gouraud')
    plt.xlabel("t [years]")
    plt.ylabel("Short rate r(t) = x(t) + f(0,t) [%]")
    clb = plt.colorbar(ticks=0.01 * np.arange(0, 21, 2))
    clb.set_label("Probability density")
    plt.ylim(-8, 13)
    plt.clim(0, 0.16)
    lgd1 = plt.plot(event_grid_plot, rate_adj_avg, "-b")
    lgd2 = plt.plot(event_grid_plot, forward_rate_plot, "--b")
    plt.legend(lgd1 + lgd2, ["E[r(t)]", "f(0,t)"])
    plt.show()

    # Density plot of discount curve
    plt.pcolormesh(XD, YD, ZD_adj, cmap=plt.colormaps['hot'], shading='gouraud')
    plt.xlabel("t [years]")
    plt.ylabel("Discount curve")
    clb = plt.colorbar(ticks=0.1 * np.arange(0, 31, 5))
    clb.set_label("Probability density")
    plt.ylim(0.1, 1.2)
    plt.clim(0, 3)
    lgd1 = plt.plot(event_grid_plot, discount_adj_avg, "-b")
    plt.legend(lgd1, ["E[D(t)]"])
    plt.show()

    # Plot Monte-Carlo scenarios
    event_grid_plot = 0.05 * np.arange(0, 501)
    n_paths = 25
    hw = sde.SDE(kappa, vol, event_grid_plot)
    rate, discount = hw.paths(0, n_paths, seed=0)

    d_curve = discount_curve.interpolation(event_grid_plot)
    discount_curve_plot = \
        misc.DiscreteFunc("discount plot", event_grid_plot, d_curve,
                          interp_scheme="cubic")
    discount_adj = sde.discount_adjustment(discount, discount_curve_plot)

    # Pseudo short rate plots
    plot1 = plt.plot(event_grid_plot, 100 * rate[:, 0], "-b", label="Path 1")
    plot2 = plt.plot(event_grid_plot, 100 * rate[:, 2], "-r", label="Path 2")
    plot3 = plt.plot(event_grid_plot, 100 * rate[:, 4], "-k", label="Path 3")
    plt.xlabel("t [years]")
    plt.ylabel("Pseudo short rate x(t) = r(t) - f(0,t) [%]")
    plots = plot1 + plot2 + plot3
    plt.legend(plots, [plot.get_label() for plot in plots], loc=3)
    plt.show()

    # Pseudo discount curve plots
    plot1 = plt.plot(event_grid_plot, discount[:, 0], "-b", label="Path 1")
    plot2 = plt.plot(event_grid_plot, discount[:, 2], "-r", label="Path 2")
    plot3 = plt.plot(event_grid_plot, discount[:, 4], "-k", label="Path 3")
    plt.xlabel("t [years]")
    plt.ylabel("Pseudo discount curve")
    plots = plot1 + plot2 + plot3
    plt.legend(plots, [plot.get_label() for plot in plots], loc=3)
    plt.show()

    # Short rate plots
    forward_rate = input.forward_rate_ext
    forward_rate_plot = forward_rate.interpolation(event_grid_plot)

    rate_adj = rate[:, 0] + forward_rate_plot
    plot1 = plt.plot(event_grid_plot, 100 * rate_adj, "-b", label="Path 1")
    rate_adj = rate[:, 2] + forward_rate_plot
    plot2 = plt.plot(event_grid_plot, 100 * rate_adj, "-r", label="Path 2")
    rate_adj = rate[:, 4] + forward_rate_plot
    plot3 = plt.plot(event_grid_plot, 100 * rate_adj, "-k", label="Path 3")
    plot4 = plt.plot(event_grid_plot, 100 * forward_rate_plot, "--k",
                     label="f(0,t)")
    plt.xlabel("t [years]")
    plt.ylabel("Short rate r(t) = x(t) + f(0,t) [%]")
    plots = plot1 + plot2 + plot3 + plot4
    plt.legend(plots, [plot.get_label() for plot in plots], loc=3)
    plt.show()

    # Discount factor plots
    plot1 = plt.plot(event_grid_plot, discount_adj[:, 0], "-b", label="Path 1")
    plot2 = plt.plot(event_grid_plot, discount_adj[:, 2], "-r", label="Path 2")
    plot3 = plt.plot(event_grid_plot, discount_adj[:, 4], "-k", label="Path 3")
    plot4 = plt.plot(event_grid_plot, d_curve, "--k", label="P(0,t)")
    plt.xlabel("t [years]")
    plt.ylabel("Discount curve")
    plots = plot1 + plot2 + plot3 + plot4
    plt.legend(plots, [plot.get_label() for plot in plots], loc=3)
    plt.show()

    # Quarterly payment dates for a 30Y bond
    event_grid = np.arange(4 * 30 + 1) / 4

    # Discount curve on event_grid
    discount_curve = discount_curve.interpolation(event_grid)
    discount_curve = \
        misc.DiscreteFunc("discount curve", event_grid,
                          discount_curve, interp_scheme="quadratic")

    # SDE object
    n_paths = 50000
    hw = sde.SDE(kappa, vol, event_grid)
    rate, discount = hw.paths(0, n_paths, seed=0)
    discount = sde.discount_adjustment(discount, discount_curve)

    # # Plot y-function
    # plt.plot(hw.event_grid, hw.y_event_grid, "-b")
    # plt.xlabel("Time")
    # plt.ylabel("y function")
    # plt.show()

    # Zero-coupon bond object
    maturity_idx = event_grid.size - 1
    bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)

    fig, ax = plt.subplots(3, 1)
    ax_count = 0
    ax_twinx = list()
    for ax_sub in ax:
        ax_twinx.append(ax_sub.twinx())
    for n in range(1, 21):
        n_payments = event_grid.size - (n + 1)
        # All "maturities" from t_{n + 1} to T inclusive
        maturity_indices = np.arange(n + 1, event_grid.size)
        # Discount curve for each pseudo rate scenario
        discount_vector = bond.price_vector(rate[n], n, maturity_indices)
        sum_discount = discount_vector.sum(axis=0)
        # Construct pseudo rate grid
        rate_max = np.max(rate[n])
        rate_min = np.min(rate[n])
        rate_interval = rate_max - rate_min
        rate_n_grid = 5
        rate_grid = np.arange(rate_n_grid) / (rate_n_grid - 1)
        rate_grid = rate_interval * rate_grid + rate_min
        # Discount curve for each pseudo rate grid point
        discount_grid = bond.price_vector(rate_grid, n, maturity_indices)
        sum_discount_grid = discount_grid.sum(axis=0)
        # Coupons of refinance bond on pseudo rate grid
        coupon_grid = np.zeros(rate_n_grid)
        for m in range(rate_n_grid):
            coupon_grid[m] = \
                misc.calc_refinancing_coupon(n_payments, sum_discount_grid[m])
        coupon_grid_save = coupon_grid
        coupon_grid = \
            misc.DiscreteFunc("Coupon grid", rate_grid,
                              coupon_grid, interp_scheme="quadratic")
        coupon_grid_int = coupon_grid.interpolation(rate[n])
        # Coupons of refinance bond for MC scenarios
        coupon_scenarios = np.zeros(n_paths)
        for m in range(n_paths):
            coupon_scenarios[m] = \
                misc.calc_refinancing_coupon(n_payments, sum_discount[m])
        print("Are all close: ",
              np.allclose(coupon_scenarios, coupon_grid_int,
                          rtol=1e-5, atol=1e-5))
        if n == 1 or n == 4 or n == 20:
            rate_n_grid_new = 100
            rate_grid_new = np.arange(rate_n_grid_new) / (rate_n_grid_new - 1)
            rate_grid_new = rate_interval * rate_grid_new + rate_min
            coupon_grid_int_new = coupon_grid.interpolation(rate_grid_new)
            if n == 20:
                ax[ax_count].set_xlabel("Pseudo short rate, "
                                        "x(t) = r(t) - f(0,t) [%]")
            ax[ax_count].hist(100 * rate[n], 100, density=True, facecolor='r')
            if n == 4:
                ax[ax_count].set_ylabel("Density of "
                                        "pseudo short rate", color="r")
            ax[ax_count].tick_params(axis="y", labelcolor="r")
            ax_twinx[ax_count].plot(100 * rate_grid_new,
                                    100 * coupon_grid_int_new, '-b')
            ax_twinx[ax_count].plot(100 * rate_grid,
                                    100 * coupon_grid_save, 'ob')
            if n == 4:
                ax_twinx[ax_count].set_ylabel("Coupon of refinancing bond [%]",
                                              color="b")
            ax_twinx[ax_count].tick_params(axis="y", labelcolor="b")
            ax[ax_count].set_xlim([-12.5, 12.5])
            ax_count += 1
    plt.show()

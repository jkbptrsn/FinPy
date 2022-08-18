import matplotlib.pyplot as plt
import numpy as np
import unittest

from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from models.hull_white import call_option
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
        # Speed of mean reversion strip.
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
        # SDE objects
        hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)
        hw_const = \
            sde.SDEConstant(kappa, vol, event_grid, int_step_size=1 / 52)
        # Pseudo rate and discount factors
        r_pseudo, d_pseudo = hw.paths(0, n_paths, seed=0, antithetic=True)
        r_pseudo_const, d_pseudo_const = \
            hw_const.paths(0, n_paths, seed=0, antithetic=True)
        # Expiry index
        expiry_idx = 5
        # Maturity index
        maturity_idx = event_grid.size - 1
        # Strike value
        strike = 0.65
        # Call option object
        call = call_option.Call(kappa, vol, discount_curve, event_grid,
                                strike, expiry_idx, maturity_idx,
                                int_step_size=1 / 52)
        # Zero-coupon bond object
        bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                       maturity_idx, int_step_size=1 / 52)
        # Threshold
        threshold = np.array([5e-5, 5e-5, 5e-5, 7e-5, 2e-4, 5e-3])
        for s in range(2, 13, 2):
            # New discount curve on event_grid
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid, discount_curve)
            # Update discount curves
            call.discount_curve = discount_curve
            call.zcbond.discount_curve = discount_curve
            bond.discount_curve = discount_curve
            # Call option price, analytical
            call_price_a = call.price(0, 0)
            # Call option price, numerical, SDE class
            # Zero-coupon bond price at expiry
            bond_price = bond.price(r_pseudo[expiry_idx], expiry_idx)
            # Call option payoff
            payoff = np.maximum(bond_price - strike, 0)
            # Discount factor
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            # Monte-Carlo estimate of call option price at time zero
            call_price_n = np.sum(discount[expiry_idx] * payoff) / n_paths
            # Relative difference
            diff = abs((call_price_a - call_price_n) / call_price_a)
            # Call option price, numerical, SDEConstant class
            # Zero-coupon bond price at expiry
            bond_price = bond.price(r_pseudo_const[expiry_idx], expiry_idx)
            # Call option payoff
            payoff = np.maximum(bond_price - strike, 0)
            # Discount factor
            discount = sde.discount_adjustment(d_pseudo_const, discount_curve)
            # Monte-Carlo estimate of call option price at time zero
            call_price_n = np.sum(discount[expiry_idx] * payoff) / n_paths
            # Relative difference
            diff_const = abs((call_price_a - call_price_n) / call_price_a)
            # print(s, call_price_a, diff, diff_const)
            self.assertTrue(diff < threshold[s // 2 - 1])
            self.assertTrue(diff_const < threshold[s // 2 - 1])

    def test_call_option_pricing_2(self):
        """Compare analytical and numerical calculation of call options
        written on zero coupon bonds.
        """
        event_grid = np.arange(11)
        # Speed of mean reversion strip -- constant kappa!
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(10),
                        0.003 * np.array([3, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve on event_grid
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # SDE objects
        n_paths = 10000
        hw = sde.SDE(kappa, vol, event_grid)
        # Pseudo rate and discount factors
        rate_pseudo, d_pseudo = \
            hw.paths(0, n_paths, seed=0, antithetic=True)
        # Call option
        maturity_idx = event_grid.size - 1
        strike = 0.65
        expiry_idx = 5
        call_1 = call_option.Call(kappa, vol, discount_curve, event_grid,
                           strike, expiry_idx, maturity_idx)
        # Zero-coupon bond
        bond = \
            zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)
        # Threshold
        threshold = np.array([3e-4, 4e-4, 4e-4, 6e-4, 8e-4,
                              1e-3, 2e-3, 6e-3, 2e-2, 9e-3])
        for s in range(2, 12, 1):
            # New discount curve on event_grid
            spot = 0.001 * s
            forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
            discount_curve = np.exp(-forward_rate * event_grid)
            discount_curve = \
                misc.DiscreteFunc("discount curve", event_grid,
                                  discount_curve, interp_scheme="linear")
            # Update discount curves
            call_1.discount_curve = discount_curve
            call_1.zcbond.discount_curve = discount_curve
            bond.discount_curve = discount_curve
            # Call option price, analytical
            call_price_a = call_1.price(0, 0)
            # Call option price, numerical
            discount = sde.discount_adjustment(d_pseudo, discount_curve)
            discount = discount[expiry_idx]
            bond_price = bond.price(rate_pseudo[expiry_idx, :], expiry_idx)
            payoff = np.maximum(bond_price - strike, 0)
            call_price_n = np.sum(discount * payoff) / n_paths
            diff = abs((call_price_a - call_price_n) / call_price_a)
            # print(s, call_price_a, call_price_n, diff)
            self.assertTrue(diff < threshold[s - 2])

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

    # Speed of mean reversion strip
    kappa = np.array([np.array([0, 10]), 0.023 * np.array([1, 1])])
    kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])

    # Volatility strip
    vol = np.array([np.array([0, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 20]),
                    np.array([0.0165, 0.0143, 0.0140, 0.0132, 0.0128, 0.0103,
                              0.0067, 0.0096, 0.0087, 0.0091, 0.0098])])
    vol = misc.DiscreteFunc("vol", vol[0], vol[1])

    # Discount curve on event_grid
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

    # Plot zero-coupon bond price curve
    time_grid_plot = 0.1 * np.arange(-50, 351)  # np.arange(301)
    plt.plot(time_grid_plot, discount_curve.interpolation(time_grid_plot))
    plt.xlabel("Time")
    plt.ylabel("Zero coupon bond price")
    plt.show()

    # Plot Monte-Carlo scenarios
    event_grid_plot = 0.01 * np.arange(0, 1001)
    n_paths = 25
    hw = sde.SDE(kappa, vol, event_grid_plot)
    rate, discount = hw.paths(0, n_paths, seed=0)
    d_curve = discount_curve.interpolation(event_grid_plot)
    f1, ax1 = plt.subplots(3, 1, sharex=True)
    for n in range(n_paths):
        ax1[0].plot(event_grid_plot, rate[:, n])
        ax1[1].plot(event_grid_plot, discount[:, n])
        ax1[2].plot(event_grid_plot, discount[:, n] * d_curve)
    ax1[2].set_xlabel("Time")
    ax1[0].set_ylabel("Short rate")
    ax1[1].set_ylabel("Discount curve")
    ax1[2].set_ylabel("Adj. discount curve")
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

    # Plot y-function
    plt.plot(hw.event_grid, hw.y_event_grid, "-b")
    plt.xlabel("Time")
    plt.ylabel("y function")
    plt.show()

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

    # Full swap rate curve
    time_grid = np.array(
        [0.025,
         0.282,
         0.531,
         0.780,
         1.029,
         1.279,
         1.530,
         1.780,
         2.026,
         2.278,
         2.533,
         2.776,
         3.025,
         3.277,
         3.532,
         3.776,
         4.025,
         4.277,
         4.537,
         4.775,
         5.024,
         5.276,
         5.533,
         5.782,
         6.032,
         6.281,
         6.533,
         6.782,
         7.028,
         7.277,
         7.532,
         7.776,
         8.025,
         8.277,
         8.531,
         8.775,
         9.024,
         9.276,
         9.531,
         9.777,
         10.026,
         10.278,
         10.535,
         10.776,
         11.026,
         11.283,
         11.532,
         11.781,
         12.030,
         12.280,
         12.531,
         12.778,
         13.027,
         13.276,
         13.531,
         13.777,
         14.026,
         14.278,
         14.533,
         14.776,
         15.026,
         15.278,
         15.538,
         15.776,
         16.025,
         16.277,
         16.534,
         16.775,
         17.024,
         17.282,
         17.531,
         17.783,
         18.029,
         18.278,
         18.530,
         18.777,
         19.026,
         19.278,
         19.529,
         19.776,
         20.025,
         20.277,
         20.529,
         20.775,
         21.024,
         21.276,
         21.528,
         21.777,
         22.026,
         22.284,
         22.533,
         22.782,
         23.031,
         23.280,
         23.530,
         23.779,
         24.028,
         24.277,
         24.529,
         24.775,
         25.024,
         25.276,
         25.528,
         25.777,
         26.027,
         26.278,
         26.530,
         26.777,
         27.026,
         27.278,
         27.535,
         27.776,
         28.025,
         28.283])

    rate_grid = np.array([
        -0.00496,
        0.00092,
        0.00638,
        0.01056,
        0.01368,
        0.01610,
        0.01797,
        0.01931,
        0.02027,
        0.02103,
        0.02162,
        0.02207,
        0.02247,
        0.02282,
        0.02315,
        0.02344,
        0.02371,
        0.02397,
        0.02422,
        0.02442,
        0.02461,
        0.02478,
        0.02493,
        0.02506,
        0.02518,
        0.02530,
        0.02542,
        0.02554,
        0.02565,
        0.02577,
        0.02589,
        0.02600,
        0.02611,
        0.02622,
        0.02634,
        0.02644,
        0.02655,
        0.02666,
        0.02676,
        0.02686,
        0.02697,
        0.02707,
        0.02718,
        0.02729,
        0.02739,
        0.02749,
        0.02759,
        0.02767,
        0.02776,
        0.02783,
        0.02789,
        0.02795,
        0.02800,
        0.02804,
        0.02808,
        0.02810,
        0.02812,
        0.02812,
        0.02812,
        0.02812,
        0.02810,
        0.02807,
        0.02804,
        0.02800,
        0.02795,
        0.02790,
        0.02784,
        0.02778,
        0.02771,
        0.02764,
        0.02756,
        0.02748,
        0.02740,
        0.02732,
        0.02723,
        0.02715,
        0.02706,
        0.02697,
        0.02688,
        0.02679,
        0.02669,
        0.02660,
        0.02651,
        0.02642,
        0.02633,
        0.02624,
        0.02615,
        0.02606,
        0.02597,
        0.02587,
        0.02579,
        0.02570,
        0.02561,
        0.02552,
        0.02543,
        0.02534,
        0.02525,
        0.02517,
        0.02508,
        0.02499,
        0.02491,
        0.02482,
        0.02473,
        0.02465,
        0.02456,
        0.02448,
        0.02439,
        0.02431,
        0.02423,
        0.02415,
        0.02406,
        0.02399,
        0.02391,
        0.02383])

    inst_forward_grid = np.array([
        -0.00435,
        0.00731,
        0.01682,
        0.02171,
        0.02498,
        0.02702,
        0.02759,
        0.02739,
        0.02717,
        0.02697,
        0.02682,
        0.02679,
        0.02697,
        0.02724,
        0.02750,
        0.02775,
        0.02800,
        0.02819,
        0.02829,
        0.02827,
        0.02816,
        0.02803,
        0.02798,
        0.02802,
        0.02814,
        0.02831,
        0.02849,
        0.02869,
        0.02890,
        0.02912,
        0.02934,
        0.02954,
        0.02974,
        0.02993,
        0.03011,
        0.03025,
        0.03037,
        0.03051,
        0.03070,
        0.03093,
        0.03121,
        0.03147,
        0.03167,
        0.03180,
        0.03187,
        0.03187,
        0.03181,
        0.03168,
        0.03149,
        0.03125,
        0.03098,
        0.03069,
        0.03037,
        0.03002,
        0.02964,
        0.02924,
        0.02881,
        0.02835,
        0.02785,
        0.02735,
        0.02681,
        0.02626,
        0.02571,
        0.02523,
        0.02474,
        0.02426,
        0.02379,
        0.02337,
        0.02295,
        0.02254,
        0.02215,
        0.02179,
        0.02145,
        0.02112,
        0.02081,
        0.02053,
        0.02026,
        0.02001,
        0.01978,
        0.01958,
        0.01939,
        0.01921,
        0.01903,
        0.01886,
        0.01869,
        0.01852,
        0.01835,
        0.01818,
        0.01802,
        0.01786,
        0.01770,
        0.01754,
        0.01739,
        0.01724,
        0.01709,
        0.01694,
        0.01680,
        0.01666,
        0.01652,
        0.01638,
        0.01625,
        0.01612,
        0.01599,
        0.01587,
        0.01576,
        0.01566,
        0.01556,
        0.01547,
        0.01538,
        0.01530,
        0.01522,
        0.01516,
        0.01509,
        0.01504])

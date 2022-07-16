import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.hull_white.sde as sde
import models.hull_white.zero_coupon_bond as zcbond
import models.hull_white.call_option as call
import utils.misc as misc


class SDE(unittest.TestCase):

    def test_y_function(self):
        """In the case of both constant speed of mean reversion and
        constant volatility, the y-function has a closed form.
        Proposition 10.1.7, L.B.G. Andersen & V.V. Piterbarg 2010.
        """
        kappa_const = 0.05
        two_kappa = 2 * kappa_const
        vol_const = 0.02
        event_grid = np.arange(0, 30, 2)
        y_analytical = \
            vol_const ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa
        # Speed of mean reversion strip
        kappa = np.array([np.arange(2), kappa_const * np.ones(2)])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(2), vol_const * np.ones(2)])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # SDE object
        hull_white = sde.SDE(kappa, vol, event_grid)
        hull_white.setup_int_grid()
        hull_white.setup_kappa_vol_y()
        for idx, y_numerical in enumerate(hull_white.y_event_grid):
            if idx >= 1:
                diff = abs(y_analytical[idx] - y_numerical) / y_analytical[idx]
                # print("test_y_function: ", idx, y_analytical[idx], diff)
                self.assertTrue(diff < 2.8e-4)

    def test_zero_coupon_bond_pricing(self):
        """Compare analytical and numerical calculation of zero-coupon
        bonds with different maturities.
        """
        event_grid = np.arange(11)
        # Speed of mean reversion strip
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([2, 1, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(10),
                        0.01 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve on event_grid
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # SDE object
        n_paths = 1000
        hull_white = sde.SDE(kappa, vol, event_grid)
        hull_white.initialization()
        rate, discount = hull_white.paths(0, n_paths, seed=0, antithetic=True)
        # Threshold
        threshold = np.array([1e-10, 4.0e-7, 3e-6, 1e-4, 3e-4,
                              5e-4, 6e-4, 6e-4, 4e-4, 3e-3, 5e-3])
        for event_idx in range(event_grid.size):
            # Analytical result
            price_a = discount_curve.values[event_idx]
            # Monte-Carlo estimate
            price_n = np.sum(discount[event_idx, :]) / n_paths
            price_n *= discount_curve.values[event_idx]
            diff = abs((price_n - price_a) / price_a)
            # print("test_zero_coupon_bond_pricing: ", event_idx, price_a, diff)
            self.assertTrue(abs(diff) < threshold[event_idx])

    def test_coupon_bond_pricing(self):
        """Compare analytical and numerical calculation of 10Y
        coupon bond with yearly coupon frequency.
        """
        coupon = 0.03
        event_grid = np.arange(11)
        # Speed of mean reversion strip
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([2, 1, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(10),
                        0.01 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve on event_grid
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # SDE object
        n_paths = 10000
        hull_white = sde.SDE(kappa, vol, event_grid)
        hull_white.initialization()
        rate, discount = hull_white.paths(0, n_paths, seed=0, antithetic=True)
        sde.discount_adjustment(event_grid, discount,
                                discount_curve, replace=True)
        price_a_c = 0
        price_a_p = 0
        price_n_c = 0
        price_n_p = 0
        for event_idx in range(1, event_grid.size):
            discount_a = discount_curve.values[event_idx]
            discount_n = np.sum(discount[event_idx, :]) / n_paths
            # Coupon
            price_a_c += coupon * discount_a
            price_n_c += coupon * discount_n
            # Principal
            if event_idx == event_grid.size - 1:
                price_a_p = discount_a
                price_n_p = discount_n
        diff_c = abs(price_n_c - price_a_c) / price_a_c
        diff_p = abs(price_n_p - price_a_p) / price_a_p
        # print("test_coupon_bond_pricing:", price_a_c, diff_c)
        # print("test_coupon_bond_pricing:", price_a_p, diff_p)
        self.assertTrue(diff_c < 2e-4)
        self.assertTrue(diff_p < 2e-3)

    def test_sde_objects(self):
        """Compare SDE objects for calculation of call options written
        on zero coupon bonds.
        """
        event_grid = np.arange(11)
        # Speed of mean reversion strip -- constant kappa!
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip -- constant vol!
        vol = np.array([np.arange(10),
                        0.003 * np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # Discount curve on event_grid
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # SDE objects
        n_paths = 1000
        hw = sde.SDE(kappa, vol, event_grid)
        hw.initialization()
        hw_const = sde.SDEConstant(kappa, vol, event_grid)
        hw_const.initialization()
        # Pseudo rate and discount factors
        rate_pseudo, discount_pseudo = hw.paths(0, n_paths, seed=0)
        rate_pseudo_const, discount_pseudo_const = \
            hw_const.paths(0, n_paths, seed=0)
        # Compare trajectories
        for n in range(n_paths):
            diff_rate = np.abs(rate_pseudo[1:, n] - rate_pseudo_const[1:, n])
            diff_rate = np.abs(diff_rate / rate_pseudo_const[1:, n])
            diff_discount = \
                np.abs(discount_pseudo[1:, n] - discount_pseudo_const[1:, n])
            diff_discount = \
                np.abs(diff_discount / discount_pseudo_const[1:, n])
            # print(n, np.max(diff_rate), np.max(diff_discount))
            self.assertTrue(np.max(diff_rate) < 3e-2)
            self.assertTrue(np.max(diff_discount) < 4e-5)
        # Compare mean and variance of pseudo short rate and discount
        # processes, respectively
        for n in range(1, event_grid.size):
            diff_rate_mean = \
                np.abs((hw.rate_mean[n] - hw_const.rate_mean[n])
                       / hw_const.rate_mean[n])
            diff_rate_variance = \
                np.abs((hw.rate_variance[n] - hw_const.rate_variance[n])
                       / hw_const.rate_variance[n])
            diff_discount_mean = \
                np.abs((hw.discount_mean[n] - hw_const.discount_mean[n])
                       / hw_const.discount_mean[n])
            diff_discount_variance = \
                np.abs((hw.discount_variance[n]
                        - hw_const.discount_variance[n])
                       / hw_const.discount_variance[n])
            diff_covariance = \
                np.abs((hw.covariance[n] - hw_const.covariance[n])
                       / hw_const.covariance[n])
            # print("Rate mean:", n, hw_const.rate_mean[n], diff_rate_mean)
            # print("Rate variance:", n, hw_const.rate_variance[n], diff_rate_variance)
            # print("Discount mean:", n, hw_const.discount_mean[n], diff_discount_mean)
            # print("Discount variance:", n, hw_const.discount_variance[n], diff_discount_variance)
            # print("Covariance:", n, hw_const.covariance[n], diff_covariance)
            self.assertTrue(diff_rate_mean[0] < 1e-10)
            self.assertTrue(diff_rate_mean[1] < 6e-5)
            self.assertTrue(diff_rate_variance < 3e-10)
            self.assertTrue(diff_discount_mean[0] < 3e-5)
            self.assertTrue(diff_discount_mean[1] < 6e-5)
            self.assertTrue(diff_discount_variance < 2e-3)
            self.assertTrue(diff_covariance < 3e-5)

    def test_call_option_pricing_1(self):
        """Compare analytical and numerical calculation of call options
        written on zero coupon bonds. Also compare SDE classes...
        """
        event_grid = np.arange(11)
        # Speed of mean reversion strip -- constant kappa!
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([1, 1, 1])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip -- constant vol!
        vol = np.array([np.arange(10),
                        0.003 * np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])])
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
        hw.initialization()
        hw_const = sde.SDEConstant(kappa, vol, event_grid)
        hw_const.initialization()
        # Pseudo rate and discount factors
        rate_pseudo, discount_pseudo = \
            hw.paths(0, n_paths, seed=0, antithetic=True)
        rate_pseudo_const, discount_pseudo_const = \
            hw_const.paths(0, n_paths, seed=0, antithetic=True)
        # Call option
        maturity_idx = event_grid.size - 1
        strike = 0.65
        expiry_idx = 5
        call_1 = call.Call(kappa, vol, discount_curve, event_grid,
                           strike, expiry_idx, maturity_idx)
        # Zero-coupon bond
        bond = \
            zcbond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)
        # Threshold
        threshold = np.array([3e-4, 4e-4, 4e-4, 5e-4, 6e-4,
                              2e-3, 3e-3, 8e-3, 2e-2, 3e-2])
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
            discount = \
                sde.discount_adjustment(event_grid, discount_pseudo,
                                        discount_curve)
            discount = discount[expiry_idx]
            bond_price = bond.price(rate_pseudo[expiry_idx, :], expiry_idx)
            payoff = np.maximum(bond_price - strike, 0)
            call_price_n = np.sum(discount * payoff) / n_paths
            diff = abs((call_price_a - call_price_n) / call_price_a)
            discount = \
                sde.discount_adjustment(event_grid, discount_pseudo_const,
                                        discount_curve)
            discount = discount[expiry_idx]
            bond_price = \
                bond.price(rate_pseudo_const[expiry_idx, :], expiry_idx)
            payoff = np.maximum(bond_price - strike, 0)
            call_price_n_const = np.sum(discount * payoff) / n_paths
            diff_const = \
                abs((call_price_a - call_price_n_const) / call_price_a)
            # print(s, call_price_a, call_price_n, diff, call_price_n_const, diff_const)
            self.assertTrue(diff < threshold[s - 2])
            self.assertTrue(diff_const < threshold[s - 2])

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
        hw.initialization()
        # Pseudo rate and discount factors
        rate_pseudo, discount_pseudo = \
            hw.paths(0, n_paths, seed=0, antithetic=True)
        # Call option
        maturity_idx = event_grid.size - 1
        strike = 0.65
        expiry_idx = 5
        call_1 = call.Call(kappa, vol, discount_curve, event_grid,
                           strike, expiry_idx, maturity_idx)
        # Zero-coupon bond
        bond = \
            zcbond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)
        # Threshold
        threshold = np.array([5e-4, 6e-4, 7e-4, 8e-4, 2e-3,
                              3e-3, 4e-3, 7e-3, 2e-2, 5e-2])
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
            discount = \
                sde.discount_adjustment(event_grid, discount_pseudo,
                                        discount_curve)
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

        # Yearly payments...
        event_grid = np.arange(4 * 30 + 1) / 4
#        event_grid = np.arange(11)

        # Discount curve on event_grid
        discount_curve = discount_curve.interpolation(event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="quadratic")

        # SDE object
        n_paths = 500000
        hull_white = sde.SDE(kappa, vol, event_grid)
        hull_white.initialization()
        rate, discount = hull_white.paths(0, n_paths, seed=0)
        sde.discount_adjustment(event_grid, discount,
                                discount_curve, replace=True)

        # Zero-coupon bond object
        maturity_idx = event_grid.size - 1
        bond = \
            zcbond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)

        for n in range(1, event_grid.size - 1):
            maturity_indices = np.arange(n + 1, event_grid.size)
#            discount_vector = bond.price_vector(rate[n], n, maturity_indices)
            print("Event idx: ", n)
#            print("Pseudo rates: \n", rate[n])
#            print("Discount curves: \n", discount_vector)
            # Yearly coupon, because payments are yearly in this example...
            n_payments = event_grid.size - n - 1
#            coupon = 0.04
#            const_payment = coupon / (1 - (1 + coupon) ** (-n_payments))
#            sum_discount = discount_vector.sum(axis=0)
#            print("# of payments: ", n_payments)
#            print("Sum discount: \n", sum_discount)
#            print("Coupon of refinance bond: ")

            rate_max = np.max(rate[n])
            rate_min = np.min(rate[n])
            rate_interval = rate_max - rate_min
            rate_n_grid = 10
            rate_grid = np.arange(rate_n_grid) / (rate_n_grid - 1)
            rate_grid = rate_interval * rate_grid + rate_min
#            print("Rate grid: \n", rate_grid)
            discount_grid = bond.price_vector(rate_grid, n, maturity_indices)
            sum_discount_grid = discount_grid.sum(axis=0)
            coupon_grid = np.zeros(rate_n_grid)
            for m in range(rate_n_grid):
                coupon_grid[m] = \
                    misc.calc_refinance_coupon(n_payments, sum_discount_grid[m])
#            print("Coupon grid: \n", coupon_grid)
            coupon_grid = \
                misc.DiscreteFunc("Coupon grid", rate_grid,
                                  coupon_grid, interp_scheme="quadratic")
            coupon_grid = coupon_grid.interpolation(rate[n])

#            for m in range(n_paths):
#                coupon = \
#                    misc.calc_refinance_coupon(n_payments, sum_discount[m])
#                print(coupon, coupon_grid[m])


if __name__ == '__main__':

    event_grid = np.arange(11)
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
    discount_curve = misc.DiscreteFunc("discount curve", time_grid,
                                       discount_curve, interp_scheme="quadratic")
    time_grid_plot = 0.1 * np.arange(0, 301)
    plt.plot(time_grid_plot, discount_curve.interpolation(time_grid_plot))
    plt.xlabel("Time")
    plt.ylabel("Zero coupon bond price")
    plt.show()
    # Plot Monte-Carlo scenarios
    event_grid_plot = 0.01 * np.arange(0, 1001)
    # SDE object
    n_paths = 25
    hull_white = sde.SDE(kappa, vol, event_grid_plot)
    hull_white.initialization()
    rate, discount = hull_white.paths(0, n_paths, seed=0)
    d_curve = discount_curve.interpolation(event_grid_plot)
    f1, ax1 = plt.subplots(3, 1, sharex=True)
    for n in range(n_paths):
        ax1[0].plot(event_grid_plot, rate[:, n])
        ax1[1].plot(event_grid_plot, discount[:, n])
        ax1[2].plot(event_grid_plot, discount[:, n] * d_curve)
    ax1[0].set_xlabel("Time")
    ax1[1].set_xlabel("Time")
    ax1[2].set_xlabel("Time")
    ax1[0].set_ylabel("Pseudo rate")
    ax1[1].set_ylabel("Pseudo discount curve")
    ax1[2].set_ylabel("Discount curve")
    plt.show()

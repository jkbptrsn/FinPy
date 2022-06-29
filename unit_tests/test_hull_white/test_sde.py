import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.hull_white.sde as sde
import models.hull_white.zero_coupon_bond as zcbond
import models.hull_white.call as call
import utils.misc as misc


class SDE(unittest.TestCase):

    def test_y_function(self):
        """In the case of both constant speed of mean reversion and
        constant volatility, the y-function has a closed form.
        """
        kappa_const = 0.05
        two_kappa = 2 * kappa_const
        vol_const = 0.02
        event_grid = np.arange(0, 30, 2)
        y_analytical = \
            vol_const ** 2 * (1 - np.exp(-two_kappa * event_grid)) / two_kappa
        # Define discount curve on event_grid
        forward_rate = 0.02 * np.ones(event_grid.size)
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # Speed of mean reversion strip
        kappa = np.array([np.arange(2), kappa_const * np.ones(2)])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(2), vol_const * np.ones(2)])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # SDE object
        hull_white = sde.SDE(kappa, vol, discount_curve, event_grid)
        hull_white.integration_grid()
        hull_white.kappa_vol_y()
        for idx, y_numerical in enumerate(hull_white.y_event_grid):
            diff = y_analytical[idx] - y_numerical
#            print("test y-function: ", abs(diff))
            self.assertTrue(abs(diff) < 1.0e-5)

    def test_zero_coupon_bond_pricing(self):
        """Compare analytical and numerical calculation of zero-coupon
        bonds with different maturities.
        """
        event_grid = np.arange(11)
        # Define discount curve on event_grid
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # Speed of mean reversion strip
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([2, 1, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(10),
                        0.01 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # SDE object
        n_paths = 20000
        np.random.seed(0)
        hull_white = sde.SDE(kappa, vol, discount_curve, event_grid)
        hull_white.initialization()
        rate, discount = hull_white.paths(0, n_paths)
        for event_idx in range(event_grid.size):
            # Analytical result
            price_a = discount_curve.values[event_idx]
            # Monte-Carlo estimate
            price_n = np.sum(discount[event_idx, :]) / n_paths
            price_n *= discount_curve.values[event_idx]
            diff = abs((price_n - price_a) / price_a)
#            print(price_a, price_n, diff)
            self.assertTrue(abs(diff) < 3.0e-3)

    def test_coupon_bond_pricing(self):
        """Compare analytical and numerical calculation of 10Y
        coupon bond with yearly coupon frequency.
        """
        coupon = 0.03
        event_grid = np.arange(11)
        # Define discount curve on event_grid
        forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = \
            misc.DiscreteFunc("discount curve", event_grid,
                              discount_curve, interp_scheme="linear")
        # Speed of mean reversion strip
        kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([2, 1, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        # Volatility strip
        vol = np.array([np.arange(10),
                        0.01 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        # SDE object
        n_paths = 20000
        np.random.seed(0)
        hull_white = sde.SDE(kappa, vol, discount_curve, event_grid)
        hull_white.initialization()
        rate, discount = hull_white.paths(0, n_paths)
        price_a_c = 0
        price_a_p = 0
        price_n_c = 0
        price_n_p = 0
        for event_idx in range(1, event_grid.size):
            discount_a = discount_curve.values[event_idx]
            discount_n = np.sum(discount[event_idx, :]) / n_paths
            discount_n *= discount_curve.values[event_idx]
            # Coupon
            price_a_c += coupon * discount_a
            price_n_c += coupon * discount_n
            # Principal
            if event_idx == event_grid.size - 1:
                price_a_p = discount_a
                price_n_p = discount_n
#        print(abs(price_n_c - price_a_c) / price_a_c)
#        print(abs(price_n_p - price_a_p) / price_a_p)
        self.assertTrue(abs(price_n_c - price_a_c) / price_a_c < 3e-3)
        self.assertTrue(abs(price_n_p - price_a_p) / price_a_p < 3e-3)


if __name__ == '__main__':

#    unittest.main()

    event_grid = np.arange(11)
    # Define discount curve on event_grid
    forward_rate = 0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
    discount_curve = np.exp(-forward_rate * event_grid)
    discount_curve = misc.DiscreteFunc("discount curve", event_grid,
                                       discount_curve, interp_scheme="linear")

    # Speed of mean reversion strip -- CONSTANT KAPPA!!!
    # Call option price only implemented for constant kappa...
    kappa = np.array([np.array([2, 3, 7]), 0.01 * np.array([1, 1, 1])])
    kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])

    # Volatility strip
    vol = np.array([np.arange(10),
                    0.003 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    vol = misc.DiscreteFunc("vol", vol[0], vol[1])

    # Plot Monte-Carlo scenarios
    event_grid_plot = 0.01 * np.arange(0, 1001)
    # SDE object
    n_paths = 10
    np.random.seed(0)
    hull_white = sde.SDE(kappa, vol, discount_curve, event_grid_plot)
    hull_white.initialization()
    rate, discount = hull_white.paths(0, n_paths)
    d_curve = discount_curve.interpolation(event_grid_plot)
    for n in range(n_paths):
        plt.plot(event_grid_plot, rate[:, n])
        plt.plot(event_grid_plot, discount[:, n])
        plt.plot(event_grid_plot, discount[:, n] * d_curve)
    plt.show()

    # European call option object
    print("European call option:")
    maturity_idx = event_grid.size - 1
    strike = 0.2
    expiry_idx = 5
    call = call.Call(kappa, vol, discount_curve, event_grid,
                     strike, expiry_idx, maturity_idx)
    # SDE object
    n_paths = 10000
    np.random.seed(0)
    hull_white = sde.SDE(kappa, vol, discount_curve, event_grid)
    hull_white.initialization()
    # Pseudo rate and discount factors
    rate_pseudo, discount_pseudo = hull_white.paths(0, n_paths)
    # Zero-coupon bond object
    bond = \
        zcbond.ZCBond(kappa, vol, discount_curve, event_grid, maturity_idx)
    for s in range(2, 12, 2):
        spot = 0.001 * s
        forward_rate = spot * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])
        discount_curve = np.exp(-forward_rate * event_grid)
        discount_curve = misc.DiscreteFunc("discount curve", event_grid,
                                           discount_curve, interp_scheme="linear")

        call.discount_curve = discount_curve
        call._zcbond.discount_curve = discount_curve
        call_price_a = call.price(0, 0)

        bond.discount_curve = discount_curve

        # Discount factors
        discount = \
            discount_pseudo[expiry_idx, :] * discount_curve.values[expiry_idx]
        bond_price = bond.price(rate_pseudo[expiry_idx, :], expiry_idx)
        payoff = np.maximum(bond_price - strike, 0)
        call_price_n = np.sum(discount * payoff) / n_paths
        diff = abs(call_price_a - call_price_n) / call_price_a
        print(spot, call_price_a, call_price_n, diff)

    unittest.main()

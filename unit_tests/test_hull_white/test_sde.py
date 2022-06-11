import math
import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.hull_white.sde as sde
import utils.misc as misc


class SDE(unittest.TestCase):

    def test_y_function(self):
        """In the case of both constant speed of mean reversion and
        constant volatility, the y-function has a closed form."""
        kappa = 0.1
        vol = 0.2
        time_grid = np.arange(0, 30, 2)
        two_kappa = 2 * kappa
        y_analytical = \
            vol ** 2 * (1 - np.exp(-two_kappa * time_grid)) / two_kappa
        forward_rate = np.array([np.arange(2), np.ones(2)])
        kappa_strip = np.array([np.arange(2), kappa * np.ones(2)])
        kappa_strip = \
            misc.DiscreteFunc("kappa", kappa_strip[0], kappa_strip[1])
        vol_strip = np.array([np.arange(2), vol * np.ones(2)])
        vol_strip = misc.DiscreteFunc("vol", vol_strip[0], vol_strip[1])
        hullwhite = sde.SDE(kappa_strip, vol_strip, forward_rate, time_grid)
        hullwhite.integration_grid()
        hullwhite.kappa_vol_y()
        for idx, event_idx in enumerate(hullwhite.int_event_idx):
            diff = y_analytical[idx] - hullwhite.y_int_grid[event_idx]
            self.assertTrue(abs(diff) < 2.0e-4)

    def test_zero_coupon_bond_pricing(self):
        """Compare analytical and numerical calculation of 10Y
        zero-coupon bonds.
        """
        forward_rate = \
            np.array([np.arange(11),
                      0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])])
        forward_rate = \
            misc.DiscreteFunc("forward rate", forward_rate[0],
                              forward_rate[1], interp_scheme="linear")
        kappa = np.array([np.array([2, 3, 7]), 0.1 * np.array([2, 1, 2])])
        kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
        vol = np.array([np.arange(10),
                        0.0004 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
        vol = misc.DiscreteFunc("vol", vol[0], vol[1])
        n_paths = 100000
        np.random.seed(0)
        event_grid = np.array([0, 10])
        for n in range(10):
            forward_rate_new = forward_rate
            for idx in range(forward_rate_new.values.size):
                forward_rate_new.values[idx] += \
                    np.random.uniform(-1, 1) * forward_rate_new.values[idx]
            hullwhite = sde.SDE(kappa, vol, forward_rate_new,
                                event_grid, int_step_size=1/52)
            hullwhite.initialization()
            rates, discounts = hullwhite.paths(0, n_paths)
            price_analytical = math.exp(hullwhite.forward_rate_contrib[-1, 1])
            price_numerical = np.sum(np.exp(discounts[-1, :])) / n_paths
            relative_diff = \
                abs((price_numerical - price_analytical) / price_analytical)
            self.assertTrue(abs(relative_diff) < 1.0e-4)


if __name__ == '__main__':

    # Plot Monte-Carlo scenarios
    forward_rate = np.array([np.arange(11),
                             0.02 * np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6])])
    forward_rate = misc.DiscreteFunc("forward rate", forward_rate[0],
                                     forward_rate[1], interp_scheme="linear")
    kappa = np.array([np.array([2, 3, 7]), 0.1 * np.array([2, 1, 2])])
    kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
    vol = np.array([np.arange(10),
                    0.0004 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    vol = misc.DiscreteFunc("vol", vol[0], vol[1])
    event_grid = 0.01 * np.arange(0, 1001)
    hullwhite = sde.SDE(kappa, vol, forward_rate, event_grid)
    hullwhite.initialization()
    n_paths = 10
    np.random.seed(0)
    rates, discounts = hullwhite.paths(0, n_paths)
    for n in range(n_paths):
        plt.plot(event_grid, rates[:, n])
        plt.plot(event_grid, np.exp(discounts[:, n]))
    plt.show()

    unittest.main()

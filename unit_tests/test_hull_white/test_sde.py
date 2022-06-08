import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.hull_white.sde as sde
import utils.misc as misc


class SDE(unittest.TestCase):

    def test_y_function(self):
        """In the case of both constant speed of mean reversion and
        constant volatility, y has a closed form."""
        kappa = 0.1
        vol = 0.2
        time_grid = np.arange(0, 30, 2)
        two_kappa = 2 * kappa
        y_analytical = \
            vol ** 2 * (1 - np.exp(-two_kappa * time_grid)) / two_kappa
        forward_rate = np.array([np.arange(2), 0.02 * np.ones(2)])
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


if __name__ == '__main__':

    forward_rate = np.array([np.arange(10),
                             0.02 * np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2])])
    forward_rate = \
        misc.DiscreteFunc("forward rate", forward_rate[0], forward_rate[1],
                          interp_scheme="linear")
    kappa = np.array([np.array([2, 3, 7]), 0.02 * np.array([5, 15, 10])])
    kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
    vol = np.array([np.arange(10),
                    0.002 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    vol = misc.DiscreteFunc("vol", vol[0], vol[1])
    # Event_grid should contain "trade date"
    event_grid = 0.01 * np.arange(0, 1001)
    hullwhite = sde.SDE(kappa, vol, forward_rate, event_grid)
    hullwhite.initialization()
    n_paths = 2
    spot_rate = 0.02
    np.random.seed(0)
    rates, discounts = hullwhite.paths(spot_rate, n_paths)
    for n in range(n_paths):
        plt.plot(event_grid, rates[:, n])
        plt.plot(event_grid, discounts[:, n])
    plt.show()
    print(rates[-1, 0], discounts[-1, 0])
    print(rates[-1, 1], discounts[-1, 1])

    unittest.main()

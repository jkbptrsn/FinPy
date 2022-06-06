import matplotlib.pyplot as plt
import numpy as np
import time
import unittest

import models.hull_white.sde as sde
import utils.misc as misc


class SDE(unittest.TestCase):

    def test_y_function(self):
        """In the case of constant speed of mean reversion and
        volatility, an analytical expression for y exists..."""
        kappa = 0.1
        two_kappa = 2 * kappa
        vol = 0.2
        time_grid = np.arange(0, 30, 2)
        y_analytical = \
            vol ** 2 * (1 - np.exp(- two_kappa * time_grid)) / two_kappa

        kappa_strip = np.array([np.arange(2), kappa * np.ones(2)])
        kappa_strip = misc.DiscreteFunc("kappa", kappa_strip[0], kappa_strip[1])
        vol_strip = np.array([np.arange(2), vol * np.ones(2)])
        vol_strip = misc.DiscreteFunc("vol", vol_strip[0], vol_strip[1])

        hullwhite = \
            sde.SDE(kappa_strip, vol_strip, time_grid, int_step_size=1/200)
        hullwhite.integration_grid()
        hullwhite.kappa_vol_y()

        for idx, event_idx in enumerate(hullwhite.int_event_idx):
            diff = y_analytical[idx] - hullwhite.y_int_grid[event_idx]
#            print(abs(diff))
            self.assertTrue(abs(diff) < 2.0e-4)

#        plt.plot(hullwhite.int_grid, hullwhite.y_int_grid, '-r')
#        plt.plot(time_grid, y_analytical, 'ob')
#        plt.show()


if __name__ == '__main__':

    vol = np.array([np.arange(10), 0.001 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    vol_strip = misc.DiscreteFunc("vol", vol[0], vol[1])
    kappa = np.array([np.array([2, 3, 7]), 0.05 * np.array([5, 15, 10])])
    kappa_strip = misc.DiscreteFunc("kappa", kappa[0], kappa[1])

    # Event_grid should contain "trade date"
    event_grid = 0.01 * np.arange(0, 1001)
    hullwhite = sde.SDE(kappa_strip, vol_strip, event_grid)

    hullwhite.initialization()
    n_paths = 2
    rate, discount = hullwhite.paths(0.02, n_paths)

    for n in range(n_paths):
        plt.plot(event_grid, rate[:, n])
        plt.plot(event_grid, discount[:, n])
    plt.show()

    start = time.time()
    hullwhite.initialization()
    end = time.time()
    print("Computation time: ", end - start)

#    print(hullwhite._rate_mean, hullwhite._rate_mean.size)
#    print(hullwhite._rate_variance, hullwhite._rate_variance.size)
#    print(hullwhite._discount_mean, hullwhite._discount_mean.size)
#    print(hullwhite._discount_variance, hullwhite._discount_variance.size)
#    print(hullwhite._covariance, hullwhite._covariance.size)

    unittest.main()

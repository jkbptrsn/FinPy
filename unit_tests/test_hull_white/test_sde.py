import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.hull_white.sde as sde


class SDE(unittest.TestCase):

    def test_integration(self):
        """Test flat integration"""
        time_grid = np.arange(10)
        values = np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])
        vol = sde.Function("vol", time_grid, values)
        integral = [0, 1, 3, 6, 7, 8, 13, 19, 25, 28]
        for idx, t in enumerate(time_grid):
            time_new = np.arange(0, t * 10 + 1) * 0.1
            self.assertTrue(abs(vol.integration(time_new) - integral[idx]) < 10-8  )


if __name__ == '__main__':

    vol = np.array([np.arange(10), np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    plt.plot(vol[0], vol[1], "bo")

    kappa = np.array([np.array([2, 3, 7]), np.array([0.5, 4, 1])])
    plt.plot(kappa[0], kappa[1], "ro")

    vol_strip = sde.Function("vol", vol[0], vol[1])
    kappa_strip = sde.Function("kappa", kappa[0], kappa[1])

    time_new = np.arange(-10, 110) * 0.1
    vol_new = vol_strip.interpolation(time_new)
    kappa_new = kappa_strip.interpolation(time_new)

    plt.plot(time_new, vol_new, "-b")
    plt.plot(time_new, kappa_new, "-r")

#    plt.show()

    # Assume event_grid contains "trade date"
    event_grid = np.arange(0, 10 + 1, 0.5)
    hullwhite = sde.SDE(kappa_strip, vol_strip, event_grid)

    hullwhite.integration_grid()

    hullwhite.initialization()

    for idx, date in enumerate(event_grid):
        print(date, hullwhite.int_event_idx[idx])
        if idx >= 1:
            int_idx1 = hullwhite.int_event_idx[idx - 1]
            int_idx2 = hullwhite.int_event_idx[idx]
            print(hullwhite.int_grid[int_idx1:int_idx2 + 1])

    unittest.main()

import numpy as np
import unittest

import models.vasicek.sde as sde
import models.vasicek.zcbond as zcbond
import utils.plots as plots


class SDE(unittest.TestCase):

    def test_getters_and_setters(self):
        """..."""
        kappa = 0.1
        mean_rate = 0.3
        vol = 0.2
        vasicek = sde.SDE(kappa, mean_rate, vol)

        self.assertEqual(vasicek.kappa, kappa)
        self.assertEqual(vasicek.mean_rate, mean_rate)
        self.assertEqual(vasicek.vol, vol)

    def test_zcbond_mc_pricing(self):
        """Test pricing of zero-coupon bond by Monte-Carlo."""
        spot = 0.02
        kappa = 1
        mean_rate = 0.04
        vol = 0.02
        vasicek = sde.SDE(kappa, mean_rate, vol)
        t_max = 10
        n_paths = 100000
        spot_vector = np.arange(-5, 6, 1) * spot
        bond = zcbond.ZCBond(kappa, mean_rate, vol, t_max)
        for s in spot_vector:
            paths = vasicek.path(s, t_max, n_paths)
            numerical = np.sum(np.exp(paths[1])) / n_paths
            analytical = bond.price(s, 0)
            relative_diff = abs((numerical - analytical) / analytical)
            self.assertTrue(relative_diff < 1.0e-3)


if __name__ == '__main__':

    spot = 0.02
    kappa = 1
    mean_rate = 0.04
    vol = 0.02
    vasicek = sde.SDE(kappa, mean_rate, vol)
    t_min = 0
    t_max = 10
    t_steps = 501
    dt = (t_max - t_min) / (t_steps - 1)
    time_grid = dt * np.arange(0, t_steps)
    path = vasicek.path_time_grid(spot, time_grid)
    plots.plot_path(time_grid, path)

    n_paths = 100000
    spot_vector = np.arange(-5, 6, 1) * spot
    bond = zcbond.ZCBond(kappa, mean_rate, vol, t_max)
    for s in spot_vector:
        paths = vasicek.path(s, t_max, n_paths)
        numerical = np.sum(np.exp(paths[1])) / n_paths
        analytical = bond.price(s, 0)
        space = " "
        if s < 0:
            space = ""
        print("Spot rate: " + space, "{:.3e}".format(s),
              "  Monte-Carlo price: ", "{:.5e}".format(numerical),
              "  Analytical price: ", "{:.5e}".format(analytical),
              "  Relative difference: ",
              "{:.3e}".format(abs((numerical - analytical) / analytical)))

    unittest.main()

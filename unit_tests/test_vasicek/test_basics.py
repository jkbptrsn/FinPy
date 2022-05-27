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

    n_paths = 1000000
    paths = vasicek.path(spot, t_max, n_paths)
    price = np.sum(np.exp(paths[1])) / n_paths

    bond = zcbond.ZCBond(kappa, mean_rate, vol, t_max)

    print(price, bond.price(spot, 0))

    unittest.main()

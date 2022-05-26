import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.vasicek.sde as sde
import utils.plots as plots


class SDE(unittest.TestCase):

    def test_getters_and_setters(self):
        kappa = 0.1
        mean_rate = 0.3
        vol = 0.2
        vasicek = sde.SDE(kappa, mean_rate, vol)

        self.assertEqual(vasicek.kappa, kappa)
        self.assertEqual(vasicek.mean_rate, mean_rate)
        self.assertEqual(vasicek.vol, vol)


if __name__ == '__main__':

    kappa = 1
    mean_rate = 0.04
    vol = 0.02
    vasicek = sde.SDE(kappa, mean_rate, vol)
    t_min = 0
    t_max = 50
    t_steps = 501
    dt = (t_max - t_min) / (t_steps - 1)
    time_grid = dt * np.arange(0, t_steps)
    path = vasicek.path_time_grid(0.02, time_grid)

    plots.plot_path(time_grid, path)

    unittest.main()

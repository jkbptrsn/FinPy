import matplotlib.pyplot as plt
import numpy as np
import unittest

import utils.misc as misc


class Integration(unittest.TestCase):

    def test_1(self):
        """Test trapezoidal integration step-by-step of piece-wise
        constant function and piece-wise linear function, respectively.
        """
        time_grid = np.arange(10)
        values = np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])
        vol_constant = misc.DiscreteFunc("vol", time_grid, values)
        vol_linear = \
            misc.DiscreteFunc("vol", time_grid, values, interp_scheme="linear")
        # Analytical integration of constant function step-by-step
        int_const = [1, 2, 3, 1, 1, 5, 6, 6, 3]
        # Analytical integration of linear function step-by-step
        int_linear = [1.5, 2.5, 2, 1, 3, 5.5, 6, 4.5, 3]
        # Trapezoidal integration step size
        int_step_size = 1.0e-5
        time_diff = np.diff(time_grid)
        for idx, dt in enumerate(time_diff):
            n_steps = round(dt / int_step_size)
            time = int_step_size * np.arange(n_steps + 1) + time_grid[idx]
            vol = vol_constant.interpolation(time)
            integral = np.sum(misc.trapezoidal(time, vol))
            self.assertTrue(abs(integral - int_const[idx]) < 1.0e-3)
            vol = vol_linear.interpolation(time)
            integral = np.sum(misc.trapezoidal(time, vol))
            self.assertTrue(abs(integral - int_linear[idx]) < 1.0e-4)

    def test_2(self):
        """Test trapezoidal integration of exponential function:
        int_0^infty exp(-x) dx = 1.
        """
        time_grid = np.arange(0, 100, 0.01)
        exp_grid = np.exp(-time_grid)
        integral = np.sum(misc.trapezoidal(time_grid, exp_grid))
        self.assertTrue(abs(integral - 1) < 1.0e-5)

    def test_3(self):
        """Test trapezoidal integration of cosine:
        int_0^{2 * pi} cos(x) dx = 0.
        """
        time_grid = np.arange(0, 2 * np.pi, 0.001)
        cos_grid = np.cos(time_grid)
        integral = np.sum(misc.trapezoidal(time_grid, cos_grid))
        self.assertTrue(abs(integral) < 2.0e-4)


if __name__ == '__main__':

    vol = np.array([np.arange(10), 0.05 * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    vol_strip = misc.DiscreteFunc("vol", vol[0], vol[1])
    plt.plot(vol[0], vol[1], "bo")
    kappa = np.array([np.array([2, 3, 7]), 0.05 * np.array([5, 15, 10])])
    kappa_strip = misc.DiscreteFunc("kappa", kappa[0], kappa[1])
    plt.plot(kappa[0], kappa[1], "ro")

    time_new = np.arange(-100, 1100) * 0.01
    vol_new = vol_strip.interpolation(time_new)
    kappa_new = kappa_strip.interpolation(time_new)

    plt.plot(time_new, vol_new, "-b")
    plt.plot(time_new, kappa_new, "-r")
    plt.show()

    unittest.main()

import numpy as np
import scipy.stats

import utils.payoffs as payoffs
import numerical_methods.finite_difference.theta as theta

import unittest


class BlackScholes(unittest.TestCase):

    def test_call_option(self):

        n_doubles = 3

        # Test convergence wrt to time and space separately

        bc_type = "Linearity"
        # bc_type = "PDE"

        rate = 0.0
        strike = 50
        vol = 0.2
        expiry = 2

        t_min = 0
        t_max = 2
        t_steps = 101
        dt = (t_max - t_min) / (t_steps - 1)

        x_min = 25
        x_max = 75
        x_steps = 101

        t_array = np.zeros(n_doubles - 1)
        x_array = np.zeros(n_doubles - 1)
        norm_array = np.zeros((3, n_doubles - 1))

        for n in range(n_doubles):
            # Set up PDE solver
            solver = theta.AndersenPiterbarg1D(x_min, x_max, x_steps, dt, theta=0.5, bc_type=bc_type)
            solver.set_drift(rate * solver.grid())
            solver.set_diffusion(vol * solver.grid())
            solver.set_rate(rate + 0 * solver.grid())

            # Terminal solution to PDE
            solver.solution = payoffs.call(solver.grid(), strike)

            solver.initialization()

            # Propagate value vector backwards in time
            for t in range(t_steps - 1):
                solver.propagation()

            # Analytical result
            value = solver.solution

            if n > 0:
                abs_diff = np.abs(value_old[1:-1] - value[1:-1][::2])

                norm_center = abs_diff[(x_steps_old - 1) // 2]
                norm_max = np.amax(abs_diff)
                # Old step size is 2 * self.dx
                norm_l2 = np.sqrt(np.sum((2 * solver.dx) * np.square(abs_diff)))

                # Data used for linear regression
                t_array[n - 1] = np.log(dt)
                x_array[n - 1] = np.log(solver.dx)
                norm_array[0, n - 1] = np.log(norm_center)
                norm_array[1, n - 1] = np.log(norm_max)
                norm_array[2, n - 1] = np.log(norm_l2)

            # Save value vector
            value_old = value

            # Update grid spacing in spatial dimension
            x_steps_old = x_steps
            x_steps = 2 * x_steps - 1

            # Update grid spacing in time dimension
            t_steps = 2 * t_steps - 1
            dt = (t_max - t_min) / (t_steps - 1)

        # Linear regression
        res_1t = scipy.stats.linregress(t_array, norm_array[0, :])
        res_2t = scipy.stats.linregress(t_array, norm_array[1, :])
        res_3t = scipy.stats.linregress(t_array, norm_array[2, :])
        res_1x = scipy.stats.linregress(x_array, norm_array[0, :])
        res_2x = scipy.stats.linregress(x_array, norm_array[1, :])
        res_3x = scipy.stats.linregress(x_array, norm_array[2, :])

        self.assertTrue(abs(res_1t.slope - 2.000274) < 1.0e-6)
        self.assertTrue(abs(res_2t.slope - 2.000248) < 1.0e-6)
        self.assertTrue(abs(res_3t.slope - 2.000146) < 1.0e-6)

        self.assertTrue(abs(res_1x.slope - 2.000274) < 1.0e-6)
        self.assertTrue(abs(res_2x.slope - 2.000248) < 1.0e-6)
        self.assertTrue(abs(res_3x.slope - 2.000146) < 1.0e-6)


class Vasicek(unittest.TestCase):

    def test_zero_coupon_bond(self):

        n_doubles = 3

        # Test convergence wrt to time and space separately

        bc_type = "Linearity"
        # bc_type = "PDE"

        vol = 0.05
        kappa = 0.1
        theta_factor = 0

        t_min = 0
        t_max = 2
        t_steps = 101
        dt = (t_max - t_min) / (t_steps - 1)

        sigma_grid = np.sqrt(vol ** 2 * (t_max - t_min))
        sigma_grid_new = np.sqrt(vol ** 2 * (1 - np.exp(-2 * kappa * (t_max - t_min))) / (2 * kappa))

        x_min = - 5 * sigma_grid
        x_max = 5 * sigma_grid
        x_steps = 101

        t_array = np.zeros(n_doubles - 1)
        x_array = np.zeros(n_doubles - 1)
        norm_array = np.zeros((3, n_doubles - 1))

        for n in range(n_doubles):
            # Set up PDE solver
            solver = theta.AndersenPiterbarg1D(x_min, x_max, x_steps, dt, bc_type=bc_type)
            solver.set_drift(kappa * (theta_factor - solver.grid()))
            solver.set_diffusion(vol + 0 * solver.grid())
            solver.set_rate(solver.grid())

            # Terminal solution to PDE
            solver.solution = 1 + 0 * solver.grid()

            solver.initialization()

            # Propagate value vector backwards in time
            for t in range(t_steps - 1):
                solver.propagation()

            # Analytical result
            value = solver.solution

            if n > 0:
                abs_diff = np.abs(value_old[1:-1] - value[1:-1][::2])

                norm_center = abs_diff[(x_steps_old - 1) // 2]
                norm_max = np.amax(abs_diff)
                # Old step size is 2 * self.dx
                norm_l2 = np.sqrt(np.sum((2 * solver.dx) * np.square(abs_diff)))

                # Data used for linear regression
                t_array[n - 1] = np.log(dt)
                x_array[n - 1] = np.log(solver.dx)
                norm_array[0, n - 1] = np.log(norm_center)
                norm_array[1, n - 1] = np.log(norm_max)
                norm_array[2, n - 1] = np.log(norm_l2)

            # Save value vector
            value_old = value

            # Update grid spacing in spatial dimension
            x_steps_old = x_steps
            x_steps = 2 * x_steps - 1

            # Update grid spacing in time dimension
            t_steps = 2 * t_steps - 1
            dt = (t_max - t_min) / (t_steps - 1)

        # Linear regression
        res_1t = scipy.stats.linregress(t_array, norm_array[0, :])
        res_2t = scipy.stats.linregress(t_array, norm_array[1, :])
        res_3t = scipy.stats.linregress(t_array, norm_array[2, :])
        res_1x = scipy.stats.linregress(x_array, norm_array[0, :])
        res_2x = scipy.stats.linregress(x_array, norm_array[1, :])
        res_3x = scipy.stats.linregress(x_array, norm_array[2, :])

        self.assertTrue(abs(res_1t.slope - 2.000072) < 1.0e-6)
        self.assertTrue(abs(res_2t.slope - 1.997120) < 1.0e-6)
        self.assertTrue(abs(res_3t.slope - 2.109779) < 1.0e-6)

        self.assertTrue(abs(res_1x.slope - 2.000072) < 1.0e-6)
        self.assertTrue(abs(res_2x.slope - 1.997120) < 1.0e-6)
        self.assertTrue(abs(res_3x.slope - 2.109779) < 1.0e-6)


if __name__ == '__main__':
    unittest.main()

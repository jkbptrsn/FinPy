import numpy as np
import scipy
import unittest

import utils.payoffs as payoffs
import models.black_scholes.call as bs_call
import numerical_methods.finite_difference.theta as theta


class BlackScholes(unittest.TestCase):
    """..."""

    def test_call_option(self):
        """..."""

        n_doubles = 3

        # Test convergence wrt to time and space separately

        model = "Black-Scholes"
        instrument = 'Call'

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

            # Reset current time
            t_current = t_max

            # Set up PDE solver
            solver = theta.SolverNew(x_min, x_max, x_steps, dt, boundary=bc_type)
            solver.initialization()

            solver.set_drift(rate * solver.grid())
            solver.set_diffusion(vol * solver.grid())
            solver.set_rate(rate + 0 * solver.grid())

            # Terminal solution to PDE
            solver.solution = payoffs.call(solver.grid(), strike)

            solver.set_bc_dt()
            solver.set_propagator()

            payoff = solver.solution.copy()

            # Propagate value vector backwards in time
            for t in range(t_steps - 1):
                solver.propagation()

            # Analytical result
            instru = bs_call.Call(rate, vol, strike, expiry)
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


if __name__ == '__main__':
    unittest.main()

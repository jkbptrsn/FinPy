import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.vasicek import sde
from models.vasicek import zero_coupon_bond as zcbond
from utils import misc

plot_results = False
print_results = True


class ZeroCouponBond(unittest.TestCase):

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = 0.1
        self.mean_rate = 0.03
        self.vol = 0.05
        # FD spatial grid.
        self.x_min = -0.4
        self.x_max = 0.6
        self.x_steps = 200
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        # FD event grid.
        self.maturity = 10
        self.t_steps = 100
        self.dt = self.maturity / (self.t_steps - 1)
        self.event_grid = self.dt * np.arange(self.t_steps)
        self.maturity_idx = self.t_steps - 1
        # Zero-coupon bond.
        self.bond = zcbond.ZCBond(self.kappa, self.mean_rate, self.vol,
                                  self.maturity_idx, self.event_grid)
        self.bond.fd_setup(self.x_grid, equidistant=True)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        self.bond.fd_solve()
        numerical = self.bond.fd.solution
        analytical = self.bond.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plt.plot(self.x_grid, relative_error, "-b")
            plt.xlabel("Short rate")
            plt.ylabel("Relative error")
            plt.pause(5)
        # Check convergence in reduced interval around strike.
        idx_min = np.argwhere(self.x_grid < -0.1)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.3)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(max_error)
        self.assertTrue(max_error < 7e-5)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of zero-coupon bond."""

        # Spor rate.
        spot = 0.02
        spot_vector = np.arange(-4, 5, 2) * spot

        # SDE object
        monte_carlo = sde.SDE(self.kappa, self.mean_rate, self.vol, self.event_grid)

        # Initialize random number generator
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate
        n_paths = 5000
        # Number of repetitions of Monte-Carlo simulation
        n_rep = 20
        for s in spot_vector:
            # Analytical result
            price_a = self.bond.price(s, 0)
            # Numerical result; no variance reduction
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = monte_carlo.paths(s, n_paths, rng=rng)
                price_n = discounts[self.maturity_idx, :].mean()
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(s, price_a, error.mean(), error.std())
#            self.assertTrue(error.mean() < 0.009 and error.std() < 0.007)
            # Numerical result; Antithetic sampling
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                rates, discounts = \
                    monte_carlo.paths(s, n_paths, rng=rng, antithetic=True)
                price_n = discounts[self.maturity_idx, :].mean()
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(s, price_a, error.mean(), error.std())
#            self.assertTrue(error.mean() < 0.006 and error.std() < 0.004)


if __name__ == '__main__':
    unittest.main()

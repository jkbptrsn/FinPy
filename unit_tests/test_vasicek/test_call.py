import unittest

import numpy as np

from models.vasicek import call_option as call
from utils import plots

plot_results = False
print_results = False

if print_results:
    print("Unit test results from: " + __name__)


class CallOption(unittest.TestCase):
    """European call option in Vasicek model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = 0.1
        self.mean_rate = 0.03
        self.vol = 0.05
        # FD spatial grid.
        self.x_min = -0.4
        self.x_max = 0.6
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        # Bond maturity.
        self.maturity = 10
        # Option expiry.
        self.expiry = 5
        # Option strike.
        self.strike = 0.8
        # FD event grid.
        self.fd_t_steps = 101
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1
        self.fd_expiry_idx = (self.fd_t_steps - 1) // 2
        # MC event grid; exact discretization.
        self.mc_t_steps = 3
        self.mc_dt = self.maturity / (self.mc_t_steps - 1)
        self.mc_event_grid = self.mc_dt * np.arange(self.mc_t_steps)
        self.mc_maturity_idx = self.mc_t_steps - 1
        self.mc_expiry_idx = (self.mc_t_steps - 1) // 2
        # MC event grid; Euler discretization.
        self.mc_euler_t_steps = 51
        self.mc_euler_dt = self.maturity / (self.mc_euler_t_steps - 1)
        self.mc_euler_event_grid = \
            self.mc_euler_dt * np.arange(self.mc_euler_t_steps)
        self.mc_euler_maturity_idx = self.mc_euler_t_steps - 1
        self.mc_euler_expiry_idx = (self.mc_euler_t_steps - 1) // 2
        # Call option.
        self.fd_call = \
            call.Call(self.kappa, self.mean_rate, self.vol, self.strike,
                      self.fd_expiry_idx, self.fd_maturity_idx,
                      self.fd_event_grid)
        self.mc_call = \
            call.Call(self.kappa, self.mean_rate, self.vol, self.strike,
                      self.mc_expiry_idx, self.mc_maturity_idx,
                      self.mc_event_grid)
        self.mc_euler_call = \
            call.Call(self.kappa, self.mean_rate, self.vol, self.strike,
                      self.mc_euler_expiry_idx, self.mc_euler_maturity_idx,
                      self.mc_euler_event_grid)

    def test_theta_method(self):
        """Finite difference pricing of European call option."""
        self.fd_call.fd_setup(self.x_grid, equidistant=True)
        self.fd_call.fd_solve()
        # Check price.
        numerical = self.fd_call.fd.solution
        analytical = self.fd_call.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.fd_call)
        # Maximum error in interval around short rate of 0.1.
        idx_min = np.argwhere(self.x_grid < -0.2)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.4)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.5f}")
        self.assertTrue(max_error < 8.3e-3)
        # Check delta.
        numerical = self.fd_call.fd.delta()
        analytical = self.fd_call.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.5f}")
        self.assertTrue(max_error < 6.0e-3)
        # Check gamma.
        numerical = self.fd_call.fd.gamma()
        analytical = self.fd_call.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.5f}")
        self.assertTrue(max_error < 8.7e-3)
        # Check theta.
        numerical = self.fd_call.fd.theta()
        analytical = self.fd_call.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.5f}")
        self.assertTrue(max_error < 5.1e-3)

    def test_monte_carlo_exact(self):
        """Monte-Carlo pricing of European call option."""
        self.mc_call.mc_exact_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = np.arange(-4, 5, 2) * spot
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 5000
        # Number of repetitions of Monte-Carlo simulation.
        n_rep = 50
        for s in spot_vector:
            # Analytical result.
            price_a = self.mc_call.price(s, 0)
            # Numerical result; no variance reduction.
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_call.mc_exact_solve(s, n_paths, rng=rng)
                price_n = self.mc_call.mc_exact.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"No variance reduction: "
                      f"Short rate = {s:5.2f}, price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.2e-2 and error.std() < 1.6e-2)
            # Numerical result; Antithetic sampling.
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_call.mc_exact_solve(s, n_paths, rng=rng,
                                            antithetic=True)
                price_n = self.mc_call.mc_exact.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"Antithetic sampling:   "
                      f"Short rate = {s:5.2f}, price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 1.8e-2 and error.std() < 1.5e-2)

    def test_monte_carlo_euler(self) -> None:
        """Monte-Carlo pricing of zero-coupon bond."""
        self.mc_euler_call.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = np.arange(-4, 5, 2) * spot
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 5000
        # Number of repetitions of Monte-Carlo simulation.
        n_rep = 50
        for s in spot_vector:
            # Analytical result.
            price_a = self.mc_euler_call.price(s, 0)
            # Numerical result; no variance reduction.
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_euler_call.mc_euler_solve(s, n_paths, rng=rng)
                price_n = self.mc_euler_call.mc_euler.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"No variance reduction: "
                      f"Short rate = {s:5.2f}, price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.1e-2 and error.std() < 1.7e-2)
            # Numerical result; Antithetic sampling.
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_euler_call.mc_euler_solve(s, n_paths, rng=rng,
                                                  antithetic=True)
                price_n = self.mc_euler_call.mc_euler.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"Antithetic sampling:   "
                      f"Short rate = {s:5.2f}, price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.2e-2 and error.std() < 1.7e-2)


if __name__ == '__main__':
    unittest.main()

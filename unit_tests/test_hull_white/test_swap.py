import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import misc_swap as misc_sw
from models.hull_white import swap
from unit_tests.test_hull_white import input
from utils import plots

plot_results = False
print_results = False


class Swap(unittest.TestCase):
    """Fixed-for-floating swap in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        self.fixed_rate = 0.02
        self.event_grid, self.fixing_schedule, self.payment_schedule = \
            misc_sw.swap_schedule(1, 5, 2, 50)
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        # Swap.
        self.time_dependence = "piecewise"
        self.swap = \
            swap.Swap(self.kappa,
                      self.vol,
                      self.discount_curve,
                      self.fixed_rate,
                      self.fixing_schedule,
                      self.payment_schedule,
                      self.event_grid,
                      self.time_dependence)
        self.swapPelsser = \
            swap.SwapPelsser(self.kappa,
                             self.vol,
                             self.discount_curve,
                             self.fixed_rate,
                             self.fixing_schedule,
                             self.payment_schedule,
                             self.event_grid,
                             self.time_dependence)

    def test_pricing(self):
        """Compare pricing functions."""
        price_1 = self.swap.price(self.x_grid, 0)
        annuity = self.swap.annuity(self.x_grid, 0)
        forward = self.swap.par_rate(self.x_grid, 0)
        price_2 = annuity * (forward - self.fixed_rate)
        if print_results:
            for x, p1, p2 in zip(self.x_grid, price_1, price_2):
                print(x, p1, p2, p1 - p2)
        self.assertTrue(np.abs(price_1 - price_2)[(self.x_steps - 1) // 2] < 1e-12)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        if print_results:
            print(self.swap.transformation)
        self.swap.fd_setup(self.x_grid, equidistant=True)
        self.swap.fd_solve()
        # Check price.
        numerical = self.swap.fd.solution
        analytical = self.swap.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.swap)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 3.0e-3)
        # Check delta.
        numerical = self.swap.fd.delta()
        analytical = self.swap.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.4e-6)
        # Check gamma.
        numerical = self.swap.fd.gamma()
        analytical = self.swap.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 2.0e-6)
        # Check theta.
        numerical = self.swap.fd.theta()
        analytical = self.swap.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 5.2e-6)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of zero-coupon bond."""
        if print_results:
            print(self.swapPelsser.transformation)
        self.swapPelsser.fd_setup(self.x_grid, equidistant=True)
        self.swapPelsser.fd_solve()
        # Check price.
        numerical = self.swapPelsser.fd.solution
        analytical = self.swapPelsser.price(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        if plot_results:
            plots.plot_price_and_greeks(self.swapPelsser)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 3.2e-3)
        # Check delta.
        numerical = self.swapPelsser.fd.delta()
        analytical = self.swapPelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.3e-3)
        # Check gamma.
        numerical = self.swapPelsser.fd.gamma()
        analytical = self.swapPelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 2.3e-3)
        # Check theta.
        numerical = self.swapPelsser.fd.theta()
        analytical = self.swapPelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 6.1e-5)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of European call option."""
        self.swap.mc_exact_setup()
        self.swap.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.swap.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.swap.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.swap.mc_exact.mc_estimate
            error_exact[idx] = self.swap.mc_exact.mc_error
            self.swap.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.swap.mc_euler.mc_estimate
            error_euler[idx] = self.swap.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Call option price")
            plt.legend()
            plt.show()
        relative_error = np.abs((price_a - numerical_exact) / price_a)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 3.1e-3)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of European call option."""
        self.swapPelsser.mc_exact_setup()
        self.swapPelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.swapPelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.swapPelsser.mc_exact_solve(s, n_paths, rng=rng,
                                            antithetic=True)
            numerical_exact[idx] = self.swapPelsser.mc_exact.mc_estimate
            error_exact[idx] = self.swapPelsser.mc_exact.mc_error
            self.swapPelsser.mc_euler_solve(s, n_paths, rng=rng,
                                            antithetic=True)
            numerical_euler[idx] = self.swapPelsser.mc_euler.mc_estimate
            error_euler[idx] = self.swapPelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Call option price")
            plt.legend()
            plt.show()
        relative_error = np.abs((price_a - numerical_exact) / price_a)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 3.1e-3)

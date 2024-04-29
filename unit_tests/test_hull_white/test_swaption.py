import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import misc_swap as misc_sw
from models.hull_white import swap
from models.hull_white import swaption
from unit_tests.test_hull_white import input
from utils import plots

plot_results = False
print_results = False


class Swaption(unittest.TestCase):
    """Payer swaption in 1-factor Hull-White model."""

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
        # Swaption.
        self.swaption = \
            swaption.Payer(self.kappa,
                           self.vol,
                           self.discount_curve,
                           self.fixed_rate,
                           self.fixing_schedule,
                           self.payment_schedule,
                           self.event_grid,
                           self.time_dependence)
        self.swaptionPelsser = \
            swaption.PayerPelsser(self.kappa,
                                  self.vol,
                                  self.discount_curve,
                                  self.fixed_rate,
                                  self.fixing_schedule,
                                  self.payment_schedule,
                                  self.event_grid,
                                  self.time_dependence)

    def test_theta_method(self):
        """Finite difference pricing of zero-coupon bond."""
        if print_results:
            print(self.swaption.transformation)
        self.swaption.fd_setup(self.x_grid, equidistant=True)
        self.swaption.fd_solve()
        # Check price.
        numerical = self.swaption.fd.solution
        analytical = self.swaption.price(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        if plot_results:
            plots.plot_price_and_greeks(self.swaption)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 2.2e-5)
        # Check delta.
        numerical = self.swaption.fd.delta()
        analytical = self.swaption.delta(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 5.8e-4)
        # Check gamma.
        numerical = self.swaption.fd.gamma()
        analytical = self.swaption.gamma(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 1.2e-1)
        # Check theta.
        numerical = self.swaption.fd.theta()
        analytical = self.swaption.theta(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 6.0e-5)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of zero-coupon bond."""
        if print_results:
            print(self.swaptionPelsser.transformation)
        self.swaptionPelsser.fd_setup(self.x_grid, equidistant=True)
        self.swaptionPelsser.fd_solve()
        # Check price.
        numerical = self.swaptionPelsser.fd.solution
        analytical = self.swaptionPelsser.price(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        if plot_results:
            plots.plot_price_and_greeks(self.swaptionPelsser)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 3.3e-4)
        # Check delta.
        numerical = self.swaptionPelsser.fd.delta()
        analytical = self.swaptionPelsser.delta(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.0e-2)
        # Check gamma.
        numerical = self.swaptionPelsser.fd.gamma()
        analytical = self.swaptionPelsser.gamma(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 4.5e-1)
        # Check theta.
        numerical = self.swaptionPelsser.fd.theta()
        analytical = self.swaptionPelsser.theta(self.x_grid, 0)
        error = np.abs(analytical - numerical)
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 9.7e-5)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of European call option."""
        self.swaption.mc_exact_setup()
        self.swaption.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.swaption.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.swaption.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.swaption.mc_exact.mc_estimate
            error_exact[idx] = self.swaption.mc_exact.mc_error
            self.swaption.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.swaption.mc_euler.mc_estimate
            error_euler[idx] = self.swaption.mc_euler.mc_error
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
        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 3.1e-4)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of European call option."""
        self.swaptionPelsser.mc_exact_setup()
        self.swaptionPelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.swaptionPelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.swaptionPelsser.mc_exact_solve(s, n_paths, rng=rng,
                                                antithetic=True)
            numerical_exact[idx] = self.swaptionPelsser.mc_exact.mc_estimate
            error_exact[idx] = self.swaptionPelsser.mc_exact.mc_error
            self.swaptionPelsser.mc_euler_solve(s, n_paths, rng=rng,
                                                antithetic=True)
            numerical_euler[idx] = self.swaptionPelsser.mc_euler.mc_estimate
            error_euler[idx] = self.swaptionPelsser.mc_euler.mc_error
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
        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 4.7e-4)

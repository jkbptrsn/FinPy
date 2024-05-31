import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import caplet as cf_hw
from unit_tests.test_hull_white import input
from utils import plots

plot_results = False
print_results = False


class CapletFloorlet(unittest.TestCase):
    """Caplet and floorlet in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        self.strike_rate = 0.02
        self.event_steps = 201
        self.payment_date = 5
        self.dt = self.payment_date / (self.event_steps - 1)
        self.event_grid = self.dt * np.arange(self.event_steps)
        self.fixing_idx = 180
        self.payment_idx = 200
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.time_dependence = "piecewise"
        # Caplet.
        self.caplet = cf_hw.Caplet(
            self.kappa, self.vol, self.discount_curve, self.strike_rate,
            self.fixing_idx, self.payment_idx, self.event_grid,
            self.time_dependence, option_type="caplet")
        self.caplet_pelsser = cf_hw.CapletPelsser(
            self.kappa, self.vol, self.discount_curve, self.strike_rate,
            self.fixing_idx, self.payment_idx, self.event_grid,
            self.time_dependence, option_type="caplet")
        # Floorlet.
        self.floorlet = cf_hw.Caplet(
            self.kappa, self.vol, self.discount_curve, self.strike_rate,
            self.fixing_idx, self.payment_idx, self.event_grid,
            self.time_dependence, option_type="floorlet")
        self.floorlet_pelsser = cf_hw.CapletPelsser(
            self.kappa, self.vol, self.discount_curve, self.strike_rate,
            self.fixing_idx, self.payment_idx, self.event_grid,
            self.time_dependence, option_type="floorlet")

    def test_theta_method_caplet(self):
        """Finite difference pricing of caplet."""
        if print_results:
            print(self.caplet.transformation)
        self.caplet.fd_setup(self.x_grid, equidistant=True)
        self.caplet.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.caplet)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.caplet.fd.solution
        analytical = self.caplet.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 3.1e-3)
        # Check delta.
        numerical = self.caplet.fd.delta()
        analytical = self.caplet.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 7.7e-4)
        # Check gamma.
        numerical = self.caplet.fd.gamma()
        analytical = self.caplet.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 9.2e-3)
        # Check theta.
        numerical = self.caplet.fd.theta()
        analytical = self.caplet.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 4.5e-6)

    def test_theta_method_caplet_pelsser(self):
        """Finite difference pricing of caplet."""
        if print_results:
            print(self.caplet_pelsser.transformation)
        self.caplet_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.caplet_pelsser.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.caplet_pelsser)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.caplet_pelsser.fd.solution
        analytical = self.caplet_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 3.3e-3)
        # Check delta.
        numerical = self.caplet_pelsser.fd.delta()
        analytical = self.caplet_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 8.2e-4)
        # Check gamma.
        numerical = self.caplet_pelsser.fd.gamma()
        analytical = self.caplet_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 9.9e-3)
        # Check theta.
        numerical = self.caplet_pelsser.fd.theta()
        analytical = self.caplet_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 4.6e-6)

    def test_theta_method_floorlet(self):
        """Finite difference pricing of floorlet."""
        if print_results:
            print(self.floorlet.transformation)
        self.floorlet.fd_setup(self.x_grid, equidistant=True)
        self.floorlet.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.floorlet)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.floorlet.fd.solution
        analytical = self.floorlet.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 6.7e-3)
        # Check delta.
        numerical = self.floorlet.fd.delta()
        analytical = self.floorlet.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.4e-3)
        # Check gamma.
        numerical = self.floorlet.fd.gamma()
        analytical = self.floorlet.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 1.7e-3)
        # Check theta.
        numerical = self.floorlet.fd.theta()
        analytical = self.floorlet.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.2e-6)

    def test_theta_method_floorlet_pelsser(self):
        """Finite difference pricing of floorlet."""
        if print_results:
            print(self.floorlet_pelsser.transformation)
        self.floorlet_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.floorlet_pelsser.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.floorlet_pelsser)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.floorlet_pelsser.fd.solution
        analytical = self.floorlet_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 6.8e-3)
        # Check delta.
        numerical = self.floorlet_pelsser.fd.delta()
        analytical = self.floorlet_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.5e-3)
        # Check gamma.
        numerical = self.floorlet_pelsser.fd.gamma()
        analytical = self.floorlet_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 1.7e-3)
        # Check theta.
        numerical = self.floorlet_pelsser.fd.theta()
        analytical = self.floorlet_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.3e-6)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of caplet."""
        self.caplet.mc_exact_setup()
        self.caplet.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        # Analytical result.
        price_a = self.caplet.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.caplet.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.caplet.mc_exact.mc_estimate
            error_exact[idx] = self.caplet.mc_exact.mc_error
            self.caplet.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.caplet.mc_euler.mc_estimate
            error_euler[idx] = self.caplet.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Caplet price")
            plt.legend()
            plt.show()
        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.1e-4)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of caplet."""
        self.floorlet_pelsser.mc_exact_setup()
        self.floorlet_pelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        # Analytical result.
        price_a = self.floorlet_pelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.floorlet_pelsser.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.floorlet_pelsser.mc_exact.mc_estimate
            error_exact[idx] = self.floorlet_pelsser.mc_exact.mc_error
            self.floorlet_pelsser.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.floorlet_pelsser.mc_euler.mc_estimate
            error_euler[idx] = self.floorlet_pelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Caplet price")
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

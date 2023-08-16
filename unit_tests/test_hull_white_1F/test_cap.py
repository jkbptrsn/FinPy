import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white_1F import cap as cf_hw
from unit_tests.test_hull_white_1F import input
from utils import plots

plot_results = False
print_results = False


class CaplFloor(unittest.TestCase):
    """Cap and floor in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        self.strike_rate = 0.02

        self.event_steps = 501
        self.payment_date = 5
        self.dt = self.payment_date / (self.event_steps - 1)
        self.event_grid = self.dt * np.arange(self.event_steps)
        self.fixing_schedule = np.array([100, 200, 300, 400])
        self.payment_schedule = np.array([200, 300, 400, 500])

        # FD spatial grid.
        self.x_min = -0.12
        self.x_max = 0.12
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.time_dependence = "piecewise"
        # Cap.
        self.cap = \
            cf_hw.Cap(self.kappa,
                      self.vol,
                      self.discount_curve,
                      self.strike_rate,
                      self.fixing_schedule,
                      self.payment_schedule,
                      self.event_grid,
                      self.time_dependence,
                      option_type="cap")
        self.cap_pelsser = \
            cf_hw.CapPelsser(self.kappa,
                             self.vol,
                             self.discount_curve,
                             self.strike_rate,
                             self.fixing_schedule,
                             self.payment_schedule,
                             self.event_grid,
                             self.time_dependence,
                             option_type="cap")
        # Floor.
        self.floor = \
            cf_hw.Cap(self.kappa,
                      self.vol,
                      self.discount_curve,
                      self.strike_rate,
                      self.fixing_schedule,
                      self.payment_schedule,
                      self.event_grid,
                      self.time_dependence,
                      option_type="floor")
        self.floor_pelsser = \
            cf_hw.CapPelsser(self.kappa,
                             self.vol,
                             self.discount_curve,
                             self.strike_rate,
                             self.fixing_schedule,
                             self.payment_schedule,
                             self.event_grid,
                             self.time_dependence,
                             option_type="floor")

    def test_theta_method_cap(self):
        """Finite difference pricing of cap."""
        if print_results:
            print(self.cap.transformation)
        self.cap.fd_setup(self.x_grid, equidistant=True)
        self.cap.fd_solve()
        # Check price.
        numerical = self.cap.fd.solution
        analytical = self.cap.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.cap)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 1.7e-3)
        # Check delta.
        numerical = self.cap.fd.delta()
        analytical = self.cap.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 5.6e-4)
        # Check gamma.
        numerical = self.cap.fd.gamma()
        analytical = self.cap.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 3.9e-3)
        # Check theta.
        numerical = self.cap.fd.theta()
        analytical = self.cap.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.5e-5)

    def test_theta_method_cap_pelsser(self):
        """Finite difference pricing of cap."""
        if print_results:
            print(self.cap_pelsser.transformation)
        self.cap_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.cap_pelsser.fd_solve()
        # Check price.
        numerical = self.cap_pelsser.fd.solution
        analytical = self.cap_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.cap_pelsser)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 1.4e-3)
        # Check delta.
        numerical = self.cap_pelsser.fd.delta()
        analytical = self.cap_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 4.1e-4)
        # Check gamma.
        numerical = self.cap_pelsser.fd.gamma()
        analytical = self.cap_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 3.3e-3)
        # Check theta.
        numerical = self.cap_pelsser.fd.theta()
        analytical = self.cap_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.3e-5)

    def test_theta_method_floor(self):
        """Finite difference pricing of floor."""
        if print_results:
            print(self.floor.transformation)
        self.floor.fd_setup(self.x_grid, equidistant=True)
        self.floor.fd_solve()
        # Check price.
        numerical = self.floor.fd.solution
        analytical = self.floor.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.floor)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 2.8e-3)
        # Check delta.
        numerical = self.floor.fd.delta()
        analytical = self.floor.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.2e-3)
        # Check gamma.
        numerical = self.floor.fd.gamma()
        analytical = self.floor.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 1.4e-3)
        # Check theta.
        numerical = self.floor.fd.theta()
        analytical = self.floor.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 3.0e-5)

    def test_theta_method_floor_pelsser(self):
        """Finite difference pricing of floor."""
        if print_results:
            print(self.floor_pelsser.transformation)
        self.floor_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.floor_pelsser.fd_solve()
        # Check price.
        numerical = self.floor_pelsser.fd.solution
        analytical = self.floor_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.floor_pelsser)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 2.2e-3)
        # Check delta.
        numerical = self.floor_pelsser.fd.delta()
        analytical = self.floor_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 7.1e-4)
        # Check gamma.
        numerical = self.floor_pelsser.fd.gamma()
        analytical = self.floor_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 1.2e-3)
        # Check theta.
        numerical = self.floor_pelsser.fd.theta()
        analytical = self.floor_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 2.8e-5)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of cap."""
        self.cap.mc_exact_setup()
        self.cap.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.cap.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.cap.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.cap.mc_exact.mc_estimate
            error_exact[idx] = self.cap.mc_exact.mc_error
            self.cap.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.cap.mc_euler.mc_estimate
            error_euler[idx] = self.cap.mc_euler.mc_error
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
        self.assertTrue(max_error < 4.9e-4)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of floor."""
        self.floor_pelsser.mc_exact_setup()
        self.floor_pelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.floor_pelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.floor_pelsser.mc_exact_solve(s, n_paths, rng=rng,
                                              antithetic=True)
            numerical_exact[idx] = self.floor_pelsser.mc_exact.mc_estimate
            error_exact[idx] = self.floor_pelsser.mc_exact.mc_error
            self.floor_pelsser.mc_euler_solve(s, n_paths, rng=rng,
                                              antithetic=True)
            numerical_euler[idx] = self.floor_pelsser.mc_euler.mc_estimate
            error_euler[idx] = self.floor_pelsser.mc_euler.mc_error
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
        self.assertTrue(max_error < 6.3e-4)


if __name__ == '__main__':
    unittest.main()

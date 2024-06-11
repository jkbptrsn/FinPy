import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import coupon_bond as bond
from models.hull_white import misc as misc_hw
from unit_tests.test_hull_white import input
from utils import cash_flows
from utils import misc
from utils import plots

plot_results = True
print_results = False


class Bond(unittest.TestCase):
    """Bond in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        # Cash flow type.
        self.cf_type = "annuity"
        # Initial time of first payment period.
        self.t_i = 0
        # Final time of last payment period.
        self.t_f = 30
        # Principal at time zero.
        self.principal = 100
        # Fixed yearly coupon.
        self.coupon = 0.05
        # Term frequency per year.
        self.frequency = 4
        # Cash flows.
        self.cash_flow_grid = cash_flows.set_payment_grid(
            self.t_i, self.t_f, self.frequency)
        self.cash_flow = cash_flows.cash_flow(
            self.coupon, self.frequency, self.cash_flow_grid, self.principal,
            self.cf_type)
        # Event grid
        event_dt = 1 / 12
        self.event_grid, self.cash_flow_schedule, _ = \
            cash_flows.set_event_grid(self.cash_flow_grid, event_dt)
        self.time_dependence = "piecewise"
        # Bond.
        self.bond = bond.Bond(
            self.kappa, self.vol, self.discount_curve, self.cash_flow_schedule,
            self.cash_flow, self.event_grid, self.time_dependence)
        self.bond_pelsser = bond.BondPelsser(
            self.kappa, self.vol, self.discount_curve, self.cash_flow_schedule,
            self.cash_flow, self.event_grid, self.time_dependence)
        # FD spatial grid.
        self.x_steps = 61
        self.x_grid = misc_hw.fd_grid(
            self.event_grid.size - 1, self.bond.vol_eg, self.event_grid,
            self.x_steps, n_stds=5, type_="hyperbolic")

    def test_theta_method(self):
        """Finite difference pricing of bond."""
        if print_results:
            print(self.bond.transformation)
        self.bond.fd_setup(self.x_grid, equidistant=False)
        self.bond.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.bond)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.bond.fd.solution
        analytical = self.bond.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 6.1e-4)
        # Check delta.
        numerical = self.bond.fd.delta()
        analytical = self.bond.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.8e-4)
        # Check gamma.
        numerical = self.bond.fd.gamma()
        analytical = self.bond.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 9.8e-4)
        # Check theta.
        numerical = self.bond.fd.theta()
        analytical = self.bond.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 2.8e-2)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of bond."""
        if print_results:
            print(self.bond_pelsser.transformation)
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=False)
        self.bond_pelsser.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.bond_pelsser)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.bond_pelsser.fd.solution
        analytical = self.bond_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 5.6e-4)
        # Check delta.
        numerical = self.bond_pelsser.fd.delta()
        analytical = self.bond_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.1e-4)
        # Check gamma.
        numerical = self.bond_pelsser.fd.gamma()
        analytical = self.bond_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 8.6e-4)
        # Check theta.
        numerical = self.bond_pelsser.fd.theta()
        analytical = self.bond_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 2.7e-2)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of bond."""
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.bond.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.bond.mc_exact.mc_estimate
            error_exact[idx] = self.bond.mc_exact.mc_error
            self.bond.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.bond.mc_euler.mc_estimate
            error_euler[idx] = self.bond.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
                fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Coupon bond price")
            plt.legend()
            plt.show()
        error = np.abs(price_a - numerical_exact) / price_a
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 4.3e-3)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of floor."""
        self.bond_pelsser.mc_exact_setup()
        self.bond_pelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.bond_pelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond_pelsser.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.bond_pelsser.mc_exact.mc_estimate
            error_exact[idx] = self.bond_pelsser.mc_exact.mc_error
            self.bond_pelsser.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.bond_pelsser.mc_euler.mc_estimate
            error_euler[idx] = self.bond_pelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
                fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Coupon bond price")
            plt.legend()
            plt.show()
        error = np.abs(price_a - numerical_exact) / price_a
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 4.3e-3)

    # @unittest.skip
    def test_monte_carlo_sobol(self):
        """Monte-Carlo pricing of European call option."""
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # Analytical result.
        analytic = self.bond.price(0, 0)
        # Number of paths per test.
        n_paths_list = (1000, 2000, 4000, 8000, 16000)
        # Number of repetitions per test.
        n_rep = 200
        # Store results.
        results_exact = np.zeros((3, len(n_paths_list)))
        results_euler = np.zeros((3, len(n_paths_list)))
        ####################################
        # Random numbers, antithetic = False
        ####################################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize random number generator.
            rng = np.random.default_rng(0)
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.bond.mc_exact_solve(
                    0, n_paths, rng=rng, antithetic=False)
                exact[n] = self.bond.mc_exact.mc_estimate
                self.bond.mc_euler_solve(
                    0, n_paths, rng=rng, antithetic=False)
                euler[n] = self.bond.mc_euler.mc_estimate
            results_exact[0, idx] = np.mean(np.abs(exact - analytic))
            results_euler[0, idx] = np.mean(np.abs(euler - analytic))
        ####################################
        # Random numbers, antithetic = False
        ####################################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize random number generator.
            rng = np.random.default_rng(0)
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.bond.mc_exact_solve(
                    0, n_paths, rng=rng, antithetic=True)
                exact[n] = self.bond.mc_exact.mc_estimate
                self.bond.mc_euler_solve(
                    0, n_paths, rng=rng, antithetic=True)
                euler[n] = self.bond.mc_euler.mc_estimate
            results_exact[1, idx] = np.mean(np.abs(exact - analytic))
            results_euler[1, idx] = np.mean(np.abs(euler - analytic))
        ################
        # Sobol sequence
        ################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize Sobol sequence generator.
            sobol_gen_exact = (
                misc.sobol_generator(2, self.bond.event_grid.size, 0))
            sobol_gen_euler = (
                misc.sobol_generator(1, self.bond.event_grid.size, 0))
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.bond.mc_exact_solve(
                    0, n_paths, sobol=True, sobol_gen=sobol_gen_exact)
                exact[n] = self.bond.mc_exact.mc_estimate
                self.bond.mc_euler_solve(
                    0, n_paths, sobol=True, sobol_gen=sobol_gen_euler)
                euler[n] = self.bond.mc_euler.mc_estimate
            results_exact[2, idx] = np.mean(np.abs(exact - analytic))
            results_euler[2, idx] = np.mean(np.abs(euler - analytic))
        if plot_results:

            print(results_exact)
            print(results_euler)

            # Plot results based on exact propagation.
            plt.plot(n_paths_list, results_exact[0, :],
                     "ob", label="RNG, trans")
            plt.plot(n_paths_list, results_euler[0, :],
                     "xb", label="RNG, euler")
            plt.plot(n_paths_list, results_exact[1, :],
                     "or", label="A-RNG, trans")
            plt.plot(n_paths_list, results_euler[1, :],
                     "xr", label="A-RNG, euler")
            plt.plot(n_paths_list, results_exact[2, :],
                     "ok", label="Sobol, trans")
            plt.plot(n_paths_list, results_euler[2, :],
                     "xk", label="Sobol, euler")
            plt.title("Coupon bond")
            plt.xlabel("Number of MC paths")
            plt.ylabel("Absolute error of MC estimate")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            file_path = "mc_convergence_coupon_bond.png"
            plt.savefig(file_path)
            # plt.show()

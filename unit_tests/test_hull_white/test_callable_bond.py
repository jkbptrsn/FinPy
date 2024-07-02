import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import callable_bond as bond
from models.hull_white import misc as misc_hw
from unit_tests.test_hull_white import input
from utils import cash_flows
from utils import misc
from utils import plots

plot_results = False
print_results = False


class FixedRate(unittest.TestCase):
    """Fixed rate callable bond in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Parameters of term structure model.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        # Cash flow type.
        self.cf_type = "deferred"
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
        # Number of 'interest only' terms.
        self.n_io_terms = 12
        # Number of terms in issuance period.
        self.n_issuance_terms = 4
        # Cash flow with issuance period.
        self.payment_grid = cash_flows.set_payment_grid_issuance(
            self.t_i, self.t_f, self.frequency, self.n_issuance_terms)
        self.cash_flow = cash_flows.cash_flow_split_issuance(
            self.coupon, self.frequency, self.payment_grid,
            self.n_issuance_terms, self.principal, self.cf_type,
            self.n_io_terms)
        # Move payment grid to "positive" times.
        self.payment_grid += self.n_issuance_terms / self.frequency
        # Event grid, and payment and deadline schedules.
        event_dt = 1 / 24
        self.event_grid, self.payment_schedule, self.deadline_schedule = \
            cash_flows.set_event_grid(self.payment_grid, event_dt)
        self.time_dependence = "piecewise"
        # Bond.
        self.bond = bond.FixedRate(
            self.kappa, self.vol, self.discount_curve, self.coupon,
            self.frequency, self.deadline_schedule, self.payment_schedule,
            self.cash_flow, self.event_grid, self.time_dependence)
        self.bond_pelsser = bond.FixedRatePelsser(
            self.kappa, self.vol, self.discount_curve, self.coupon,
            self.frequency, self.deadline_schedule, self.payment_schedule,
            self.cash_flow, self.event_grid, self.time_dependence)
        # FD spatial grids.
        self.x_steps = 101
        self.x_grid = misc_hw.fd_grid(
            self.bond.event_grid.size - 1, self.bond.vol_eg,
            self.bond.event_grid, self.x_steps, type_="hyperbolic")

    def test_theta_method(self):
        """Finite difference pricing of non-callable bond."""
        if print_results:
            print(self.bond.transformation)
        self.bond.callable_bond = False
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
        self.assertTrue(max_error < 3.3e-4)
        # Check delta.
        numerical = self.bond.fd.delta()
        analytical = self.bond.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.9e-4)
        # Check gamma.
        numerical = self.bond.fd.gamma()
        analytical = self.bond.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 5.3e-4)
        # Check theta.
        numerical = self.bond.fd.theta()
        analytical = self.bond.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.7e-2)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of non-callable bond."""
        if print_results:
            print(self.bond_pelsser.transformation)
        self.bond_pelsser.callable_bond = False
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
        self.assertTrue(max_error < 3.1e-4)
        # Check delta.
        numerical = self.bond_pelsser.fd.delta()
        analytical = self.bond_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.6e-4)
        # Check gamma.
        numerical = self.bond_pelsser.fd.gamma()
        analytical = self.bond_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 4.9e-4)
        # Check theta.
        numerical = self.bond_pelsser.fd.theta()
        analytical = self.bond_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 1.6e-2)

    def test_theta_method_compare(self):
        """Finite difference pricing of callable bond."""
        if print_results:
            print(self.bond.transformation)
            print(self.bond_pelsser.transformation)
        self.bond.fd_setup(self.x_grid, equidistant=False)
        self.bond.fd_solve()
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=False)
        self.bond_pelsser.fd_solve()
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical1 = self.bond.fd.solution
        numerical2 = self.bond_pelsser.fd.solution
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
        self.assertTrue(max_error < 2.6e-5)
        # Check delta.
        numerical1 = self.bond.fd.delta()[idx_min:idx_max + 1]
        numerical2 = self.bond_pelsser.fd.delta()[idx_min:idx_max + 1]
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error)
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
        self.assertTrue(max_error < 4.8e-4)
        # Check gamma.
        numerical1 = self.bond.fd.gamma()[idx_min:idx_max + 1]
        numerical2 = self.bond_pelsser.fd.gamma()[idx_min:idx_max + 1]
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error)
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        self.assertTrue(max_error < 1.5e-3)
        # Check theta.
        numerical1 = self.bond.fd.theta()[idx_min:idx_max + 1]
        numerical2 = self.bond_pelsser.fd.theta()[idx_min:idx_max + 1]
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error)
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
        self.assertTrue(max_error < 5.3e-3)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of non-callable bond."""
        self.bond.callable_bond = False
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        # Analytical result.
        price_a = self.bond.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.bond.mc_exact.mc_estimate
            error_exact[idx] = self.bond.mc_exact.mc_error
            self.bond.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
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
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
        relative_error = np.abs((price_a - numerical_exact) / price_a)
        max_error = np.max(relative_error)
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 1.1e-2)

    def test_monte_carlo_compare(self):
        """Monte-Carlo pricing of callable bond."""
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # Spot rate.
        spot_start = 5
        spot_steps = 5
        spot_vector = self.x_grid[spot_start:-spot_start:spot_steps]
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        # FD is the "analytical" result.
        self.bond.fd_setup(self.x_grid, equidistant=False)
        self.bond.fd_solve()
        price_a = self.bond.fd.solution
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.bond.mc_exact.mc_estimate
            error_exact[idx] = self.bond.mc_exact.mc_error
            self.bond.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.bond.mc_euler.mc_estimate
            error_euler[idx] = self.bond.mc_euler.mc_error
        if plot_results:
            plt.plot(self.x_grid, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
                fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
        price_a_tmp = price_a[spot_start:-spot_start:spot_steps]
        relative_error = np.abs((price_a_tmp - numerical_exact) / price_a_tmp)
        max_error = np.max(relative_error)
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.0e-3)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of non-callable bond."""
        self.bond_pelsser.callable_bond = False
        self.bond_pelsser.mc_exact_setup()
        self.bond_pelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        # Analytical result.
        price_a = self.bond_pelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond_pelsser.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = (
                self.bond_pelsser.mc_exact.mc_estimate)
            error_exact[idx] = self.bond_pelsser.mc_exact.mc_error
            self.bond_pelsser.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = (
                self.bond_pelsser.mc_euler.mc_estimate)
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
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
        relative_error = np.abs((price_a - numerical_exact) / price_a)
        max_error = np.max(relative_error)
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 1.1e-2)

    def test_monte_carlo_pelsser_compare(self):
        """Monte-Carlo pricing of callable bond."""
        self.bond_pelsser.mc_exact_setup()
        self.bond_pelsser.mc_euler_setup()
        # Spot rate.
        spot_start = 5
        spot_steps = 5
        spot_vector = self.x_grid[spot_start:-spot_start:spot_steps]
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        # FD is the "analytical" result.
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=False)
        self.bond_pelsser.fd_solve()
        price_a = self.bond_pelsser.fd.solution
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond_pelsser.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = (
                self.bond_pelsser.mc_exact.mc_estimate)
            error_exact[idx] = self.bond_pelsser.mc_exact.mc_error
            self.bond_pelsser.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = (
                self.bond_pelsser.mc_euler.mc_estimate)
            error_euler[idx] = self.bond_pelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(self.x_grid, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
                fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
        price_a_tmp = price_a[spot_start:-spot_start:spot_steps]
        relative_error = np.abs((price_a_tmp - numerical_exact) / price_a_tmp)
        max_error = np.max(relative_error)
        if print_results:
            print("max error: ", max_error)
        self.assertTrue(max_error < 2.0e-3)

    def test_oas(self):
        """Compare equidistant and non-equidistant FD grids."""
        x_steps_eq = 301
        x_grid_eq = misc_hw.fd_grid(
            self.bond.event_grid.size - 1, self.bond.vol_eg,
            self.bond.event_grid, x_steps_eq, type_="equidistant")
        self.bond.fd_setup(x_grid_eq, equidistant=True)
        self.bond.fd_solve()
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=False)
        self.bond_pelsser.fd_solve()
        if plot_results:
            plt.plot(x_grid_eq, np.zeros(x_grid_eq.size),
                     '.b', label="Equidistant")
            plt.plot(self.x_grid, np.zeros(self.x_grid.size),
                     'xr', label="Non-equidistant")
            plt.xlabel("Grid")
            plt.legend()
            plt.show()
        for price in range(90, 105 + 1, 2):
            oas_eq = 1.0e4 * self.bond.oas_calc(price)
            oas_noneq = 1.0e4 * self.bond_pelsser.oas_calc(price)
            diff = abs(oas_eq - oas_noneq)
            if print_results:
                print(f"Bond price = {price:3.0f}  "
                      f"OAS, eq = {oas_eq:7.2f}  "
                      f"OAS, non-eq = {oas_noneq:7.2f}  "
                      f"OAS diff = {diff:4.2f}")
            self.assertTrue(diff < 0.23)

    @unittest.skip
    def test_monte_carlo_sobol(self):
        """Monte-Carlo pricing of callable bond."""
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # FD numerical result.
        self.bond.fd_setup(self.x_grid, equidistant=False)
        self.bond.fd_solve()
        analytic = self.bond.fd.solution[(self.x_grid.size - 1) // 2]
        # Number of paths per test.
        n_paths_list = (1000, 2000, 4000, 8000, 16000)
        # Number of repetitions per test.
        n_rep = 1000
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
            plt.title("Callable bond")
            plt.xlabel("Number of MC paths")
            plt.ylabel("Absolute error of MC estimate")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            file_path = "mc_convergence_callable.png"
            plt.savefig(file_path)
            # plt.show()


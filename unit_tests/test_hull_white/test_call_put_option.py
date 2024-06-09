import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import european_option as option
from models.hull_white import misc_european_option as misc_ep
from unit_tests.test_hull_white import input
from utils import data_types
from utils import misc
from utils import plots

plot_results = False
print_results = False


class VFunction(unittest.TestCase):
    """Calculation of v- and dv_dt-functions."""

    def setUp(self) -> None:
        # Number of years
        self.n_years = 15
        # Event dates in year fractions.
        self.event_grid = np.arange(self.n_years + 1)
        # Speed of mean reversion strip.
        self.kappa_scalar = 0.02
        self.kappa_vector1 = self.kappa_scalar * np.ones(self.event_grid.size)
        self.kappa1 = data_types.DiscreteFunc(
            "kappa1", self.event_grid, self.kappa_vector1)
        # Volatility strip.
        self.vol_scalar = 0.05
        self.vol_vector1 = self.vol_scalar * np.ones(self.event_grid.size)
        # Constant vol strip.
        self.vol1 = data_types.DiscreteFunc(
            "vol1", self.event_grid, self.vol_vector1)
        self.vol_vector2 = np.zeros(self.event_grid.size)
        for idx in range(self.event_grid.size):
            self.vol_vector2[idx] = (idx % 4 + 1) * self.vol_vector1[idx]
        # Piecewise-constant vol strip.
        self.vol2 = data_types.DiscreteFunc(
            "vol2", self.event_grid, self.vol_vector2)
        # Discount curve.
        self.discount_curve = data_types.DiscreteFunc(
            "discount", self.event_grid, np.ones(self.event_grid.size))
        # Bond maturity.
        self.maturity_idx = self.event_grid.size - 1
        self.maturity = self.event_grid[self.maturity_idx]
        # Option expiry.
        self.expiry_idx = (self.event_grid.size - 1) // 2 + 1
        self.expiry = self.event_grid[self.expiry_idx]
        # Option strike price.
        self.strike = 0.8
        # Call objects.
        self.call_constant1 = option.EuropeanOption(
            self.kappa1, self.vol1, self.discount_curve, self.strike,
            self.expiry_idx, self.maturity_idx, self.event_grid, "constant",
            1 / 5)
        self.call_piecewise2 = option.EuropeanOption(
            self.kappa1, self.vol2, self.discount_curve, self.strike,
            self.expiry_idx, self.maturity_idx, self.event_grid, "piecewise",
            1 / 5)
        self.call_general1 = option.EuropeanOption(
            self.kappa1, self.vol1, self.discount_curve, self.strike,
            self.expiry_idx, self.maturity_idx, self.event_grid, "general",
            1 / 100)
        self.call_general2 = option.EuropeanOption(
            self.kappa1, self.vol2, self.discount_curve, self.strike,
            self.expiry_idx, self.maturity_idx, self.event_grid, "general",
            1 / 100)

    def test_constant(self):
        """Constant vol strip."""
        v_constant = misc_ep.v_constant(
            self.kappa_scalar, self.vol_scalar, self.expiry_idx,
            self.event_grid)
        v_constant = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_constant1.zcbond.g_eg, v_constant)
        dv_dt_constant = misc_ep.dv_dt_constant(
            self.kappa_scalar, self.vol_scalar, self.expiry_idx,
            self.event_grid)
        dv_dt_constant = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_constant1.zcbond.g_eg, dv_dt_constant)
        v_piecewise = misc_ep.v_piecewise(
            self.kappa_scalar, self.call_constant1.vol_eg, self.expiry_idx,
            self.event_grid)
        v_piecewise = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_constant1.zcbond.g_eg, v_piecewise)
        dv_dt_piecewise = misc_ep.dv_dt_piecewise(
            self.kappa_scalar, self.call_constant1.vol_eg, self.expiry_idx,
            self.event_grid)
        dv_dt_piecewise = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_constant1.zcbond.g_eg, dv_dt_piecewise)
        v_general = misc_ep.v_general(
            self.call_general1.zcbond.int_grid,
            self.call_general1.zcbond.int_event_idx,
            self.call_general1.zcbond.int_kappa_step_ig,
            self.call_general1.zcbond.vol_ig, self.expiry_idx)
        v_general = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_constant1.zcbond.g_eg, v_general)
        dv_dt_general = misc_ep.dv_dt_general(
            self.call_general1.zcbond.int_event_idx,
            self.call_general1.zcbond.int_kappa_step_ig,
            self.call_general1.zcbond.vol_ig, self.expiry_idx)
        dv_dt_general = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_constant1.zcbond.g_eg, dv_dt_general)
        if plot_results:
            event_grid_plot = self.event_grid[:self.expiry_idx + 1]
            plt.plot(event_grid_plot, v_constant, "-b", label="Constant")
            plt.plot(event_grid_plot, v_piecewise, "or", label="Piecewise")
            plt.plot(event_grid_plot, v_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("v-function")
            plt.legend()
            plt.show()
            event_grid_plot = self.event_grid[:self.expiry_idx + 1]
            plt.plot(event_grid_plot, dv_dt_constant, "-b", label="Constant")
            plt.plot(event_grid_plot, dv_dt_piecewise, "or", label="Piecewise")
            plt.plot(event_grid_plot, dv_dt_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("dv_dt-function")
            plt.legend()
            plt.show()
        for idx, (v1, v2, v3) in \
                enumerate(zip(v_constant, v_piecewise, v_general)):
            if idx != v_constant.size - 1:
                diff1 = abs(abs(v1 - v2) / v1)
                diff2 = abs(abs(v1 - v3) / v1)
                if print_results:
                    print(diff1, diff2)
                self.assertTrue(diff1 < 1.0e-15)
                self.assertTrue(diff2 < 1.4e-8)
        for idx, (v1, v2, v3) in \
                enumerate(zip(dv_dt_constant, dv_dt_piecewise, dv_dt_general)):
            if idx != v_constant.size - 1:
                diff1 = abs(abs(v1 - v2) / v1)
                diff2 = abs(abs(v1 - v3) / v1)
                if print_results:
                    print(diff1, diff2)
                self.assertTrue(diff1 < 1.0e-15)
                self.assertTrue(diff2 < 1.0e-15)

    def test_piecewise(self):
        """Piecewise constant vol strip."""
        v_piecewise = misc_ep.v_piecewise(
            self.kappa_scalar, self.call_piecewise2.vol_eg, self.expiry_idx,
            self.event_grid)
        v_piecewise = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_piecewise2.zcbond.g_eg, v_piecewise)
        dv_dt_piecewise = misc_ep.dv_dt_piecewise(
            self.kappa_scalar, self.call_piecewise2.vol_eg, self.expiry_idx,
            self.event_grid)
        dv_dt_piecewise = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx,
            self.call_piecewise2.zcbond.g_eg, dv_dt_piecewise)
        v_general = misc_ep.v_general(
            self.call_general2.zcbond.int_grid,
            self.call_general2.zcbond.int_event_idx,
            self.call_general2.zcbond.int_kappa_step_ig,
            self.call_general2.zcbond.vol_ig, self.expiry_idx)
        v_general = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx, self.call_general2.zcbond.g_eg,
            v_general)
        dv_dt_general = misc_ep.dv_dt_general(
            self.call_general2.zcbond.int_event_idx,
            self.call_general2.zcbond.int_kappa_step_ig,
            self.call_general2.zcbond.vol_ig, self.expiry_idx)
        dv_dt_general = misc_ep.v_function(
            self.expiry_idx, self.maturity_idx, self.call_general2.zcbond.g_eg,
            dv_dt_general)
        if plot_results:
            event_grid_plot = self.event_grid[:self.expiry_idx + 1]
            plt.plot(event_grid_plot, v_piecewise, "or", label="Piecewise")
            plt.plot(event_grid_plot, v_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("v-function")
            plt.legend()
            plt.show()
            event_grid_plot = self.event_grid[:self.expiry_idx + 1]
            plt.plot(event_grid_plot, dv_dt_piecewise, "or", label="Piecewise")
            plt.plot(event_grid_plot, dv_dt_general, "xk", label="General")
            plt.xlabel("Time")
            plt.ylabel("dv_dt-function")
            plt.legend()
            plt.show()
        for idx, (v1, v2) in \
                enumerate(zip(v_piecewise, v_general)):
            if idx != v_general.size - 1:
                diff = abs(abs(v1 - v2) / v1)
                if print_results:
                    print(diff)
                self.assertTrue(diff < 4.8e-3)
        for idx, (v1, v2) in \
                enumerate(zip(dv_dt_piecewise, dv_dt_general)):
            if idx != v_general.size - 1:
                diff = abs(abs(v1 - v2) / v1)
                if print_results:
                    print(diff)
                self.assertTrue(diff < 6.7e-9)


class Call(unittest.TestCase):
    """European call option in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 30
        # FD event grid.
        self.fd_t_steps = 361
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1
        # Option expiry.
        self.fd_expiry_idx = (self.fd_t_steps - 1) // 2 + 1
        self.expiry = self.fd_event_grid[self.fd_expiry_idx]
        # Option strike price.
        self.strike = 0.8
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.int_step_factor = 2
        self.int_step_size = self.fd_dt / self.int_step_factor
        # Call option.
        self.time_dependence = "piecewise"
        self.call = option.EuropeanOption(
            self.kappa, self.vol, self.discount_curve, self.strike,
            self.fd_expiry_idx, self.fd_maturity_idx, self.fd_event_grid,
            self.time_dependence, self.int_step_size)
        self.callPelsser = option.EuropeanOptionPelsser(
            self.kappa, self.vol, self.discount_curve, self.strike,
            self.fd_expiry_idx, self.fd_maturity_idx, self.fd_event_grid,
            self.time_dependence, self.int_step_size)

    def test_theta_method(self):
        """Finite difference pricing of European call option."""
        if print_results:
            print(self.call.transformation)
        self.call.fd_setup(self.x_grid, equidistant=True)
        self.call.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.call)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.call.fd.solution
        analytical = self.call.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.5f}")
        self.assertTrue(max_error < 9.5e-3)
        # Check delta.
        numerical = self.call.fd.delta()
        analytical = self.call.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.5f}")
        self.assertTrue(max_error < 6.6e-3)
        # Check gamma.
        numerical = self.call.fd.gamma()
        analytical = self.call.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.5f}")
        self.assertTrue(max_error < 4.6e-3)
        # Check theta.
        numerical = self.call.fd.theta()
        analytical = self.call.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.5f}")
        self.assertTrue(max_error < 2.5e-4)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of European call option."""
        if print_results:
            print(self.callPelsser.transformation)
        self.callPelsser.fd_setup(self.x_grid, equidistant=True)
        self.callPelsser.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.callPelsser)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.callPelsser.fd.solution
        analytical = self.callPelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.5f}")
        self.assertTrue(max_error < 9.0e-3)
        # Check delta.
        numerical = self.callPelsser.fd.delta()
        analytical = self.callPelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.5f}")
        self.assertTrue(max_error < 6.4e-3)
        # Check gamma.
        numerical = self.callPelsser.fd.gamma()
        analytical = self.callPelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.5f}")
        self.assertTrue(max_error < 4.7e-3)
        # Check theta.
        numerical = self.callPelsser.fd.theta()
        analytical = self.callPelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.5f}")
        self.assertTrue(max_error < 2.8e-4)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of European call option."""
        self.call.mc_exact_setup()
        self.call.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.call.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.call.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.call.mc_exact.mc_estimate
            error_exact[idx] = self.call.mc_exact.mc_error
            self.call.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.call.mc_euler.mc_estimate
            error_euler[idx] = self.call.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
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
        self.assertTrue(max_error < 4.9e-2)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of European call option."""
        self.callPelsser.mc_exact_setup()
        self.callPelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.callPelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.callPelsser.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.callPelsser.mc_exact.mc_estimate
            error_exact[idx] = self.callPelsser.mc_exact.mc_error
            self.callPelsser.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.callPelsser.mc_euler.mc_estimate
            error_euler[idx] = self.callPelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
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
        self.assertTrue(max_error < 4.9e-2)

    @unittest.skip
    def test_monte_carlo_sobol(self):
        """Monte-Carlo pricing of European call option."""
        self.call.mc_exact_setup()
        self.call.mc_euler_setup()
        # Analytical result.
        analytic = self.call.price(0, 0)
        # Number of paths per test.
        n_paths_list = (1000, 4000, 16000, 64000)
        # Number of repetitions per test.
        n_rep = 250
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
                self.call.mc_exact_solve(
                    0, n_paths, rng=rng, antithetic=False)
                exact[n] = self.call.mc_exact.mc_estimate
                self.call.mc_euler_solve(
                    0, n_paths, rng=rng, antithetic=False)
                euler[n] = self.call.mc_euler.mc_estimate
            results_exact[0, idx] = (
                np.mean(np.abs((exact - analytic) / analytic)))
            results_euler[0, idx] = (
                np.mean(np.abs((euler - analytic) / analytic)))
        ####################################
        # Random numbers, antithetic = False
        ####################################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize random number generator.
            rng = np.random.default_rng(0)
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.call.mc_exact_solve(
                    0, n_paths, rng=rng, antithetic=True)
                exact[n] = self.call.mc_exact.mc_estimate
                self.call.mc_euler_solve(
                    0, n_paths, rng=rng, antithetic=True)
                euler[n] = self.call.mc_euler.mc_estimate
            results_exact[1, idx] = (
                np.mean(np.abs((exact - analytic) / analytic)))
            results_euler[1, idx] = (
                np.mean(np.abs((euler - analytic) / analytic)))
        ################
        # Sobol sequence
        ################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize Sobol sequence generator.
            sobol_gen_exact = (
                misc.sobol_generator(2, self.call.event_grid.size, 0))
            sobol_gen_euler = (
                misc.sobol_generator(1, self.call.event_grid.size, 0))
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.call.mc_exact_solve(
                    0, n_paths, sobol=True, sobol_gen=sobol_gen_exact)
                exact[n] = self.call.mc_exact.mc_estimate
                self.call.mc_euler_solve(
                    0, n_paths, sobol=True, sobol_gen=sobol_gen_euler)
                euler[n] = self.call.mc_euler.mc_estimate
            results_exact[2, idx] = (
                np.mean(np.abs((exact - analytic) / analytic)))
            results_euler[2, idx] = (
                np.mean(np.abs((euler - analytic) / analytic)))
        if plot_results:
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
            plt.title("European call option")
            plt.xlabel("Number of MC paths")
            plt.ylabel("Relative error of MC estimate")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            file_path = "mc_convergence_call.png"
            plt.savefig(file_path)
            # plt.show()


class Put(unittest.TestCase):
    """European put option in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve
        # Bond maturity.
        self.maturity = 30
        # FD event grid.
        self.fd_t_steps = 361
        self.fd_dt = self.maturity / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_maturity_idx = self.fd_t_steps - 1
        # Option expiry.
        self.fd_expiry_idx = (self.fd_t_steps - 1) // 2 + 1
        self.expiry = self.fd_event_grid[self.fd_expiry_idx]
        # Option strike price.
        self.strike = 0.8
        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.int_step_factor = 2
        self.int_step_size = self.fd_dt / self.int_step_factor
        # Call option.
        self.time_dependence = "piecewise"
        self.put = option.EuropeanOption(
            self.kappa, self.vol, self.discount_curve, self.strike,
            self.fd_expiry_idx, self.fd_maturity_idx, self.fd_event_grid,
            self.time_dependence, self.int_step_size, option_type="Put")

        self.putPelsser = option.EuropeanOptionPelsser(
            self.kappa, self.vol, self.discount_curve, self.strike,
            self.fd_expiry_idx, self.fd_maturity_idx, self.fd_event_grid,
            self.time_dependence, self.int_step_size, option_type="Put")

    def test_theta_method(self):
        """Finite difference pricing of European call option."""
        if print_results:
            print(self.put.transformation)
        self.put.fd_setup(self.x_grid, equidistant=True)
        self.put.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.put)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.put.fd.solution
        analytical = self.put.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.5f}")
        self.assertTrue(max_error < 7.5e-4)
        # Check delta.
        numerical = self.put.fd.delta()
        analytical = self.put.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.5f}")
        self.assertTrue(max_error < 2.4e-3)
        # Check gamma.
        numerical = self.put.fd.gamma()
        analytical = self.put.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.5f}")
        self.assertTrue(max_error < 3.5e-3)
        # Check theta.
        numerical = self.put.fd.theta()
        analytical = self.put.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.5f}")
        self.assertTrue(max_error < 4.0e-5)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of European call option."""
        if print_results:
            print(self.putPelsser.transformation)
        self.putPelsser.fd_setup(self.x_grid, equidistant=True)
        self.putPelsser.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(self.putPelsser)
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        # Check price.
        numerical = self.putPelsser.fd.solution
        analytical = self.putPelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.5f}")
        self.assertTrue(max_error < 8.1e-4)
        # Check delta.
        numerical = self.putPelsser.fd.delta()
        analytical = self.putPelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.5f}")
        self.assertTrue(max_error < 2.3e-3)
        # Check gamma.
        numerical = self.putPelsser.fd.gamma()
        analytical = self.putPelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.5f}")
        self.assertTrue(max_error < 3.6e-3)
        # Check theta.
        numerical = self.putPelsser.fd.theta()
        analytical = self.putPelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.5f}")
        self.assertTrue(max_error < 4.0e-5)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of European call option."""
        self.put.mc_exact_setup()
        self.put.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.put.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.put.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.put.mc_exact.mc_estimate
            error_exact[idx] = self.put.mc_exact.mc_error
            self.put.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.put.mc_euler.mc_estimate
            error_euler[idx] = self.put.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
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
        self.assertTrue(max_error < 8.7e-3)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of European call option."""
        self.putPelsser.mc_exact_setup()
        self.putPelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.putPelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.putPelsser.mc_exact_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.putPelsser.mc_exact.mc_estimate
            error_exact[idx] = self.putPelsser.mc_exact.mc_error
            self.putPelsser.mc_euler_solve(
                s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.putPelsser.mc_euler.mc_estimate
            error_euler[idx] = self.putPelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(
                spot_vector, numerical_exact, yerr=error_exact,
                fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(
                spot_vector, numerical_euler, yerr=error_euler,
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
        self.assertTrue(max_error < 8.1e-3)

    @unittest.skip
    def test_monte_carlo_sobol(self):
        """Monte-Carlo pricing of European put option."""
        self.put.mc_exact_setup()
        self.put.mc_euler_setup()
        # Analytical result.
        analytic = self.put.price(0, 0)
        # Number of paths per test.
        n_paths_list = (1000, 4000, 16000, 64000)
        # Number of repetitions per test.
        n_rep = 250
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
                self.put.mc_exact_solve(
                    0, n_paths, rng=rng, antithetic=False)
                exact[n] = self.put.mc_exact.mc_estimate
                self.put.mc_euler_solve(
                    0, n_paths, rng=rng, antithetic=False)
                euler[n] = self.put.mc_euler.mc_estimate
            results_exact[0, idx] = (
                np.mean(np.abs((exact - analytic) / analytic)))
            results_euler[0, idx] = (
                np.mean(np.abs((euler - analytic) / analytic)))
        ####################################
        # Random numbers, antithetic = False
        ####################################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize random number generator.
            rng = np.random.default_rng(0)
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.put.mc_exact_solve(
                    0, n_paths, rng=rng, antithetic=True)
                exact[n] = self.put.mc_exact.mc_estimate
                self.put.mc_euler_solve(
                    0, n_paths, rng=rng, antithetic=True)
                euler[n] = self.put.mc_euler.mc_estimate
            results_exact[1, idx] = (
                np.mean(np.abs((exact - analytic) / analytic)))
            results_euler[1, idx] = (
                np.mean(np.abs((euler - analytic) / analytic)))
        ################
        # Sobol sequence
        ################
        for idx, n_paths in enumerate(n_paths_list):
            # Initialize Sobol sequence generator.
            sobol_gen_exact = (
                misc.sobol_generator(2, self.put.event_grid.size, 0))
            sobol_gen_euler = (
                misc.sobol_generator(1, self.put.event_grid.size, 0))
            exact = np.zeros(n_rep)
            euler = np.zeros(n_rep)
            for n in range(n_rep):
                self.put.mc_exact_solve(
                    0, n_paths, sobol=True, sobol_gen=sobol_gen_exact)
                exact[n] = self.put.mc_exact.mc_estimate
                self.put.mc_euler_solve(
                    0, n_paths, sobol=True, sobol_gen=sobol_gen_euler)
                euler[n] = self.put.mc_euler.mc_estimate
            results_exact[2, idx] = (
                np.mean(np.abs((exact - analytic) / analytic)))
            results_euler[2, idx] = (
                np.mean(np.abs((euler - analytic) / analytic)))
        if plot_results:
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
            plt.title("European put option")
            plt.xlabel("Number of MC paths")
            plt.ylabel("Relative error of MC estimate")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            file_path = "mc_convergence_put.png"
            plt.savefig(file_path)
            # plt.show()

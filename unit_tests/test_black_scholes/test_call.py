import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.black_scholes import european_option as option
from models.black_scholes import binary_option as binary
from utils import plots

plot_results = False
print_results = False

if print_results:
    print("Unit test results from: " + __name__)


class EuropeanCall(unittest.TestCase):
    """European call option in Black-Scholes model."""

    def setUp(self) -> None:
        # Model parameters.
        self.rate = 0.05
        self.vol = 0.2
        # Spot prices.
        self.spot = np.arange(2, 200, 2)
        # FD spatial grid.
        self.x_min = 2
        self.x_max = 200
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        # Option strike.
        self.strike = 50
        # Current time.
        self.time = 0
        self.time_idx = 0
        # Option expiry.
        self.expiry = 5
        # Event grid used in decomposition.
        self.expiry_idx = 2
        self.event_grid = np.array([self.time, self.expiry / 2, self.expiry])
        # FD event grid.
        self.fd_t_steps = 201
        self.fd_dt = self.expiry / (self.fd_t_steps - 1)
        self.fd_event_grid = self.fd_dt * np.arange(self.fd_t_steps)
        self.fd_expiry_idx = self.fd_t_steps - 1
        # MC event grid; exact discretization.
        self.mc_t_steps = 3
        self.mc_dt = self.expiry / (self.mc_t_steps - 1)
        self.mc_event_grid = self.mc_dt * np.arange(self.mc_t_steps)
        self.mc_expiry_idx = self.mc_t_steps - 1
        # MC event grid; Euler discretization.
        self.mc_euler_t_steps = 51
        self.mc_euler_dt = self.expiry / (self.mc_euler_t_steps - 1)
        self.mc_euler_event_grid = \
            self.mc_euler_dt * np.arange(self.mc_euler_t_steps)
        self.mc_euler_expiry_idx = self.mc_euler_t_steps - 1
        # Call option.
        self.fd_call = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.fd_expiry_idx,
            self.fd_event_grid, type_="Call")
        self.mc_call = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.mc_expiry_idx,
            self.mc_event_grid, type_="Call")
        self.mc_euler_call = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.mc_euler_expiry_idx,
            self.mc_euler_event_grid, type_="Call")

    def test_decomposition(self) -> None:
        """Decompose call option price.

        The payoff of European call option can be decomposed into
        payoffs of European asset-or-nothing and cash-or-nothing call
        options written on same underlying:
            (S - K)^+ = S * I_{S > K} - K * I_{S > K}.
        """
        c = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid,
            type_="Call")
        b_asset = binary.BinaryAssetCall(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid)
        b_cash = binary.BinaryCashCall(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid)
        price_c = c.price(self.spot, self.time_idx)
        price_ba = b_asset.price(self.spot, self.time_idx)
        price_bc = self.strike * b_cash.price(self.spot, self.time_idx)
        call_decomposed = price_ba - price_bc
        diff = np.abs(price_c - call_decomposed)
        if print_results:
            print(np.max(diff))
        self.assertTrue(np.max(diff) < 5.0e-14)
        if plot_results:
            plt.plot(self.spot, c.payoff(self.spot), "-k", label="Call payoff")
            plt.plot(self.spot, price_c, "-ob", label="Call")
            plt.plot(self.spot, price_ba, "-r", label="Asset-or-nothing call")
            plt.plot(self.spot, price_bc, "-g", label="Cash-or-nothing call")
            plt.plot(self.spot, call_decomposed, "-y", label="Composition")
            plt.title("Call option, Black-Scholes model")
            plt.xlabel("Stock price")
            plt.ylabel("Option price")
            plt.legend()
            plt.show()

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
        idx_min = np.argwhere(self.x_grid < 30)[-1][0]
        idx_max = np.argwhere(self.x_grid < 80)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.5f}")
        self.assertTrue(max_error < 8.0e-5)
        # Check delta.
        numerical = self.fd_call.fd.delta()
        analytical = self.fd_call.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.5f}")
        self.assertTrue(max_error < 4.0e-4)
        # Check gamma.
        numerical = self.fd_call.fd.gamma()
        analytical = self.fd_call.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.5f}")
        self.assertTrue(max_error < 3.4e-4)
        # Check theta.
        numerical = self.fd_call.fd.theta()
        analytical = self.fd_call.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.5f}")
        self.assertTrue(max_error < 2.0e-3)

    def test_monte_carlo_exact(self):
        """Monte-Carlo pricing of European call option."""
        self.mc_call.mc_exact_setup()
        # Spot stock price.
        spot_vector = np.arange(30, 81, 10)
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
            price_n = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_call.mc_exact_solve(s, n_paths, rng=rng)
                price_n[rep] = self.mc_call.mc_exact.mc_estimate
            error = np.abs((price_n - price_a) / price_a)
            if print_results:
                print(f"No variance reduction: "
                      f"Stock price = {s:5.2f}, "
                      f"option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.8e-2 and error.std() < 2.1e-2)
            # Numerical result; Antithetic sampling.
            price_n_anti = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_call.mc_exact_solve(
                    s, n_paths, rng=rng, antithetic=True)
                price_n_anti[rep] = self.mc_call.mc_exact.mc_estimate
            error = np.abs((price_n_anti - price_a) / price_a)
            if print_results:
                print(f"Antithetic sampling:   "
                      f"Stock price = {s:5.2f}, "
                      f"option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.7e-2 and error.std() < 2.2e-2)
            if plot_results:
                y, x, _ = plt.hist(price_n)
                plt.vlines(price_a, 0, y.max(), colors="r")
                plt.xlabel("Price")
                plt.ylabel("Count")
                plt.pause(0.5)
                plt.clf()

    def test_monte_carlo_euler(self) -> None:
        """Monte-Carlo pricing of European call option."""
        self.mc_euler_call.mc_euler_setup()
        # Spot stock price.
        spot_vector = np.arange(30, 81, 10)
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
            price_n = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_euler_call.mc_euler_solve(s, n_paths, rng=rng)
                price_n[rep] = self.mc_euler_call.mc_euler.mc_estimate
            error = abs((price_n - price_a) / price_a)
            if print_results:
                print(f"No variance reduction: "
                      f"Stock price = {s:5.2f}, "
                      f"option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 3.1e-2 and error.std() < 2.2e-2)
            # Numerical result; Antithetic sampling.
            price_n_anti = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_euler_call.mc_euler_solve(
                    s, n_paths, rng=rng, antithetic=True)
                price_n_anti[rep] = self.mc_euler_call.mc_euler.mc_estimate
            error = abs((price_n_anti - price_a) / price_a)
            if print_results:
                print(f"Antithetic sampling:   "
                      f"Stock price = {s:5.2f}, "
                      f"option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 3.4e-2 and error.std() < 2.2e-2)
            if plot_results:
                y, x, _ = plt.hist(price_n)
                plt.vlines(price_a, 0, y.max(), colors="r")
                plt.xlabel("Price")
                plt.ylabel("Count")
                plt.pause(2)
                plt.clf()

    def test_monte_carlo_plot(self) -> None:
        """Monte-Carlo pricing of European call option."""
        self.mc_call.mc_exact_setup()
        self.mc_euler_call.mc_euler_setup()
        # Spot stock price.
        spot_vector = np.arange(30, 81, 10)
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        p_a = np.zeros(spot_vector.shape)
        p_n_exact = np.zeros(spot_vector.shape)
        p_n_euler = np.zeros(spot_vector.shape)
        p_n_exact_error = np.zeros(spot_vector.shape)
        p_n_euler_error = np.zeros(spot_vector.shape)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 2000
        for idx, s in enumerate(spot_vector):
            p_a[idx] = self.mc_call.price(s, 0)
            self.mc_call.mc_exact_solve(s, n_paths, rng)
            p_n_exact[idx] = self.mc_call.mc_exact.mc_estimate
            p_n_exact_error[idx] = self.mc_call.mc_exact.mc_error
            self.mc_euler_call.mc_euler_solve(s, n_paths, rng)
            p_n_euler[idx] = self.mc_euler_call.mc_euler.mc_estimate
            p_n_euler_error[idx] = self.mc_euler_call.mc_euler.mc_error
        # Plot error bars corresponding to 95%-confidence intervals.
        p_n_exact_error *= 1.96
        p_n_euler_error *= 1.96
        if plot_results:
            plt.plot(spot_vector, p_a, "-b")
            plt.errorbar(spot_vector, p_n_exact, p_n_exact_error,
                         linestyle="none", marker="o", color="b", capsize=5)
            plt.title(f"95% confidence intervals ({n_paths} samples)")
            plt.xlabel("Spot rate")
            plt.ylabel("Price")
            plt.show()


class AmericanCall(unittest.TestCase):
    """Numerical examples in Longstaff & Schwartz 2001."""

    def setUp(self) -> None:
        # Short term interest rate.
        self.rate = 0.06
        # Strike price of put.
        self.strike = 40

        # Volatility of returns.
        self.vol1 = 0.2
        self.vol2 = 0.4

        # Event grids used in FD pricing.
        self.frequency_fd = 100
        self.event_grid_fd1 = (
            np.arange(1 * self.frequency_fd + 1) / self.frequency_fd)
        self.event_grid_fd2 = (
            np.arange(2 * self.frequency_fd + 1) / self.frequency_fd)

        # Event grids used in MC pricing.
        self.frequency_mc = 100
        self.skip = 2
        self.event_grid_mc1 = (
            np.arange(1 * self.frequency_mc + 1) / self.frequency_mc)
        self.event_grid_mc2 = (
            np.arange(2 * self.frequency_mc + 1) / self.frequency_mc)

        # List of exercise indices in ascending order.
        self.exercise_indices_1 = (
            np.arange(self.skip, 1 * self.frequency_mc + 1, self.skip))
        self.exercise_indices_2 = (
            np.arange(self.skip, 2 * self.frequency_mc + 1, self.skip))

        # Spatial grid used in FD pricing.
        self.x_grid = np.arange(603 + 1) / 3
        self.x_grid = self.x_grid[1:]

        # Number of MC paths.
        self.n_paths = 50000

        # Option objects.
        self.cFDa11 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_1,
            self.event_grid_fd1, type_="Call")
        self.cMCa11 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_1,
            self.event_grid_mc1, type_="Call")
        self.c11 = option.EuropeanOption(
            self.rate, self.vol1, self.strike, self.event_grid_fd1.size - 1,
            self.event_grid_fd1, type_="Call")

        self.cFDa12 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_2,
            self.event_grid_fd2, type_="Call")
        self.cMCa12 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_2,
            self.event_grid_mc2, type_="Call")
        self.c12 = option.EuropeanOption(
            self.rate, self.vol1, self.strike, self.event_grid_fd2.size - 1,
            self.event_grid_fd2, type_="Call")

        self.cFDa21 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_1,
            self.event_grid_fd1, type_="Call")
        self.cMCa21 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_1,
            self.event_grid_mc1, type_="Call")
        self.c21 = option.EuropeanOption(
            self.rate, self.vol2, self.strike, self.event_grid_fd1.size - 1,
            self.event_grid_fd1, type_="Call")

        self.cFDa22 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_2,
            self.event_grid_fd2, type_="Call")
        self.cMCa22 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_2,
            self.event_grid_mc2, type_="Call")
        self.c22 = option.EuropeanOption(
            self.rate, self.vol2, self.strike, self.event_grid_fd2.size - 1,
            self.event_grid_fd2, type_="Call")

    def test_pricing(self):
        """Compare prices."""
        self.cFDa11.fd_setup(self.x_grid, equidistant=True)
        self.cMCa11.mc_exact_setup()
        self.c11.mc_exact_setup()
        self.cFDa11.fd_solve()
        analytical11 = self.c11.price(self.x_grid, 0)

        self.cFDa12.fd_setup(self.x_grid, equidistant=True)
        self.cMCa12.mc_exact_setup()
        self.c12.mc_exact_setup()
        self.cFDa12.fd_solve()
        analytical12 = self.c12.price(self.x_grid, 0)

        self.cFDa21.fd_setup(self.x_grid, equidistant=True)
        self.cMCa21.mc_exact_setup()
        self.c21.mc_exact_setup()
        self.cFDa21.fd_solve()
        analytical21 = self.c21.price(self.x_grid, 0)

        self.cFDa22.fd_setup(self.x_grid, equidistant=True)
        self.cMCa22.mc_exact_setup()
        self.c22.mc_exact_setup()
        self.cFDa22.fd_solve()
        analytical22 = self.c22.price(self.x_grid, 0)

        counter = 0
        if print_results:
            print("  S  CF European  MC European     "
                  "MC error  FD American  MC American")

        rng = np.random.default_rng(0)

        for y in (36, 38, 40, 42, 44):

            self.c11.mc_exact_solve(y, self.n_paths, rng=rng, antithetic=True)
            c11_mean = self.c11.mc_exact.mc_estimate
            c11_error = self.c11.mc_exact.mc_error

            self.cMCa11.mc_exact_solve(
                y, self.n_paths, rng=rng, antithetic=True)
            ca11_mc = self.cMCa11.mc_exact.mc_estimate

            self.c12.mc_exact_solve(y, self.n_paths, rng=rng, antithetic=True)
            c12_mean = self.c12.mc_exact.mc_estimate
            c12_error = self.c12.mc_exact.mc_error

            self.cMCa12.mc_exact_solve(
                y, self.n_paths, rng=rng, antithetic=True)
            ca12_mc = self.cMCa12.mc_exact.mc_estimate

            self.c21.mc_exact_solve(y, self.n_paths, rng=rng, antithetic=True)
            c21_mean = self.c21.mc_exact.mc_estimate
            c21_error = self.c21.mc_exact.mc_error

            self.cMCa21.mc_exact_solve(
                y, self.n_paths, rng=rng, antithetic=True)
            ca21_mc = self.cMCa21.mc_exact.mc_estimate

            self.c22.mc_exact_solve(y, self.n_paths, rng=rng, antithetic=True)
            c22_mean = self.c22.mc_exact.mc_estimate
            c22_error = self.c22.mc_exact.mc_error

            self.cMCa22.mc_exact_solve(
                y, self.n_paths, rng=rng, antithetic=True)
            ca22_mc = self.cMCa22.mc_exact.mc_estimate

            for x, ca, c in \
                    zip(self.x_grid, self.cFDa11.fd.solution, analytical11):
                if abs(x - y) < 1.e-6:
                    counter += 1
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{c:11.3f}  "
                              f"{c11_mean:11.3f}  "
                              f"{c11_error:11.3f}  "
                              f"{ca:11.3f}  "
                              f"{ca11_mc:11.3f}  ")
            for x, ca, c in \
                    zip(self.x_grid, self.cFDa12.fd.solution, analytical12):
                if abs(x - y) < 1.e-6:
                    counter += 1
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{c:11.3f}  "
                              f"{c12_mean:11.3f}  "
                              f"{c12_error:11.3f}  "
                              f"{ca:11.3f}  "
                              f"{ca12_mc:11.3f}  ")
            for x, ca, c in \
                    zip(self.x_grid, self.cFDa21.fd.solution, analytical21):
                if abs(x - y) < 1.e-6:
                    counter += 1
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{c:11.3f}  "
                              f"{c21_mean:11.3f}  "
                              f"{c21_error:11.3f}  "
                              f"{ca:11.3f}  "
                              f"{ca21_mc:11.3f}  ")
            for x, ca, c in \
                    zip(self.x_grid, self.cFDa22.fd.solution, analytical22):
                if abs(x - y) < 1.e-6:
                    counter += 1
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{c:11.3f}  "
                              f"{c22_mean:11.3f}  "
                              f"{c22_error:11.3f}  "
                              f"{ca:11.3f}  "
                              f"{ca22_mc:11.3f}  ")
            print("")

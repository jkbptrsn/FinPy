import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.black_scholes import european_option as option
from models.black_scholes import binary_option as binary
from numerics.mc import lsm
from utils import plots

plot_results = False
print_results = True

if print_results:
    print("Unit test results from: " + __name__)


class CallOption(unittest.TestCase):
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
            self.rate, self.vol, self.strike, self.expiry_idx,
            self.event_grid, type_="Call")
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
            plt.pause(2)
            plt.clf()

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
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_call.mc_exact_solve(s, n_paths, rng=rng)
                price_n = self.mc_call.mc_exact.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"No variance reduction: "
                      f"Stock price = {s:5.2f}, option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.8e-2 and error.std() < 2.1e-2)
            # Numerical result; Antithetic sampling.
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_call.mc_exact_solve(
                    s, n_paths, rng=rng, antithetic=True)
                price_n = self.mc_call.mc_exact.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"Antithetic sampling:   "
                      f"Stock price = {s:5.2f}, option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 2.7e-2 and error.std() < 2.2e-2)

    def test_monte_carlo_euler(self) -> None:
        """Monte-Carlo pricing of zero-coupon bond."""
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
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_euler_call.mc_euler_solve(s, n_paths, rng=rng)
                price_n = self.mc_euler_call.mc_euler.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"No variance reduction: "
                      f"Stock price = {s:5.2f}, option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 3.1e-2 and error.std() < 2.2e-2)
            # Numerical result; Antithetic sampling.
            error = np.zeros(n_rep)
            for rep in range(n_rep):
                self.mc_euler_call.mc_euler_solve(
                    s, n_paths, rng=rng, antithetic=True)
                price_n = self.mc_euler_call.mc_euler.mc_estimate
                error[rep] += abs((price_n - price_a) / price_a)
            if print_results:
                print(f"Antithetic sampling:   "
                      f"Stock price = {s:5.2f}, option price = {price_a:2.3f}, "
                      f"error mean = {error.mean():2.5f}, "
                      f"error std = {error.std():2.5f}")
            self.assertTrue(error.mean() < 3.4e-2 and error.std() < 2.2e-2)

    def test_monte_carlo_plot(self) -> None:
        """Monte-Carlo pricing of zero-coupon bond."""
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


# TODO: Check test case!
class LongstaffSchwartz(unittest.TestCase):
    """Numerical examples in Longstaff-Schwartz article.

    See Longstaff & Schwartz 2001.
    """

    def setUp(self) -> None:
        # Finite-difference prices of American put option in
        # Black-Scholes model, Tabel 1.
        self.fd_american = \
            (4.478, 4.840, 7.101, 8.508,
             3.250, 3.745, 6.148, 7.670,
             2.314, 2.885, 5.312, 6.920,
             1.617, 2.212, 4.582, 6.248,
             1.110, 1.690, 3.948, 5.647)
        # Threshold for testing.
        self.threshold = 6e-3
        # Short term interest rate.
        self.rate = 0.06
        # Strike price of put.
        self.strike = 40

        # Volatility of returns.
        self.vol1 = 0.2
        self.vol2 = 0.4

        # Event grids used in FD pricing.
        self.frequency_fd = 500
        self.event_grid_fd1 = \
            np.arange(int(1 * self.frequency_fd) + 1) / self.frequency_fd
        self.event_grid_fd2 = \
            np.arange(int(2 * self.frequency_fd) + 1) / self.frequency_fd

        # Event grids used in MC pricing.
        self.frequency_mc = 500
        self.skip = 10
        self.event_grid_mc1 = \
            np.arange(int(1 * self.frequency_mc) + 1) / self.frequency_mc
        self.event_grid_mc2 = \
            np.arange(int(2 * self.frequency_mc) + 1) / self.frequency_mc

        self.exercise_indices_mc1 = np.arange(1 * self.frequency_mc, 0, -self.skip)
        self.exercise_indices_mc2 = np.arange(2 * self.frequency_mc, 0, -self.skip)

        # Spatial grid used in FD pricing.
        self.x_grid = np.arange(801) / 4
        self.x_grid = self.x_grid[1:]

        # Number of MC paths.
        self.n_paths = 30000

        self.pFDa11 = \
            option.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_fd1)

        self.pMCa11 = \
            option.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_mc1)

        self.p11 = \
            option.EuropeanOption(self.rate,
                    self.vol1,
                    self.strike,
                    self.event_grid_fd1.size - 1,
                    self.event_grid_fd1, type_="Call")

        self.pFDa12 = \
            option.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_fd2)

        self.pMCa12 = \
            option.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_mc2)

        self.p12 = \
            option.EuropeanOption(self.rate,
                    self.vol1,
                    self.strike,
                    self.event_grid_fd2.size - 1,
                    self.event_grid_fd2, type_="Call")

        self.pFDa21 = \
            option.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_fd1)

        self.pMCa21 = \
            option.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_mc1)

        self.p21 = \
            option.EuropeanOption(self.rate,
                    self.vol2,
                    self.strike,
                    self.event_grid_fd1.size - 1,
                    self.event_grid_fd1, type_="Call")

        self.pFDa22 = \
            option.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_fd2)

        self.pMCa22 = \
            option.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_mc2)

        self.p22 = \
            option.EuropeanOption(self.rate,
                    self.vol2,
                    self.strike,
                    self.event_grid_fd2.size - 1,
                    self.event_grid_fd2, type_="Call")

    def test_pricing(self):
        """..."""
        self.pFDa11.fd_setup(self.x_grid, equidistant=True)
        self.pMCa11.mc_exact_setup()
        self.p11.mc_exact_setup()
        self.pFDa11.fd_solve()
        analytical11 = self.p11.price(self.x_grid, 0)

        self.pFDa12.fd_setup(self.x_grid, equidistant=True)
        self.pMCa12.mc_exact_setup()
        self.p12.mc_exact_setup()
        self.pFDa12.fd_solve()
        analytical12 = self.p12.price(self.x_grid, 0)

        self.pFDa21.fd_setup(self.x_grid, equidistant=True)
        self.pMCa21.mc_exact_setup()
        self.p21.mc_exact_setup()
        self.pFDa21.fd_solve()
        analytical21 = self.p21.price(self.x_grid, 0)

        self.pFDa22.fd_setup(self.x_grid, equidistant=True)
        self.pMCa22.mc_exact_setup()
        self.p22.mc_exact_setup()
        self.pFDa22.fd_solve()
        analytical22 = self.p22.price(self.x_grid, 0)

        counter = 0
        if print_results:
            print("  S  FD European  MC European     "
                  "MC error  FD American  MC American")
        for y in (36, 38, 40, 42, 44):

            self.p11.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p11_mean = self.p11.mc_exact.mc_estimate
            p11_error = self.p11.mc_exact.mc_error

            self.pMCa11.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa11.mc_exact_solve()
            pa11_mc = lsm.american_option(self.pMCa11)

            self.p12.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p12_mean = self.p12.mc_exact.mc_estimate
            p12_error = self.p12.mc_exact.mc_error

            self.pMCa12.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa12.mc_exact_solve()
            pa12_mc = lsm.american_option(self.pMCa12)

            self.p21.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p21_mean = self.p21.mc_exact.mc_estimate
            p21_error = self.p21.mc_exact.mc_error

            self.pMCa21.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa21.mc_exact_solve()
            pa21_mc = lsm.american_option(self.pMCa21)

            self.p22.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p22_mean = self.p22.mc_exact.mc_estimate
            p22_error = self.p22.mc_exact.mc_error

            self.pMCa22.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa22.mc_exact_solve()
            pa22_mc = lsm.american_option(self.pMCa22)

            for x, pa, p in \
                    zip(self.x_grid, self.pFDa11.fd.solution, analytical11):
                if abs(x - y) < 1.e-6:
                    diff = self.fd_american[counter] - pa
                    counter += 1
#                    self.assertTrue(abs(diff) < self.threshold)
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{p:11.3f}  "
                              f"{p11_mean:11.3f}  "
                              f"{p11_error:11.3f}  "
                              f"{pa:11.3f}  "
                              f"{pa11_mc:11.3f}  ")
            for x, pa, p in \
                    zip(self.x_grid, self.pFDa12.fd.solution, analytical12):
                if abs(x - y) < 1.e-6:
                    diff = self.fd_american[counter] - pa
                    counter += 1
#                    self.assertTrue(abs(diff) < self.threshold)
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{p:11.3f}  "
                              f"{p12_mean:11.3f}  "
                              f"{p12_error:11.3f}  "
                              f"{pa:11.3f}  "
                              f"{pa12_mc:11.3f}  ")
            for x, pa, p in \
                    zip(self.x_grid, self.pFDa21.fd.solution, analytical21):
                if abs(x - y) < 1.e-6:
                    diff = self.fd_american[counter] - pa
                    counter += 1
#                    self.assertTrue(abs(diff) < self.threshold)
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{p:11.3f}  "
                              f"{p21_mean:11.3f}  "
                              f"{p21_error:11.3f}  "
                              f"{pa:11.3f}  "
                              f"{pa21_mc:11.3f}  ")
            for x, pa, p in \
                    zip(self.x_grid, self.pFDa22.fd.solution, analytical22):
                if abs(x - y) < 1.e-6:
                    diff = self.fd_american[counter] - pa
                    counter += 1
#                    self.assertTrue(abs(diff) < self.threshold)
                    if print_results:
                        print(f"{int(x):3}  "
                              f"{p:11.3f}  "
                              f"{p22_mean:11.3f}  "
                              f"{p22_error:11.3f}  "
                              f"{pa:11.3f}  "
                              f"{pa22_mc:11.3f}  ")
            print("")
        if plot_results:
            plots.plot_price_and_greeks(self.pFDa11)

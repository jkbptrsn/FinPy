import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import european_option as option
from models.black_scholes import binary_option as binary
from numerics.mc import lsm
from utils import plots

plot_results = False
print_results = False

if print_results:
    print("Unit test results from: " + __name__)


class PutOption(unittest.TestCase):
    """European call option in Black-Scholes model."""

    def setUp(self) -> None:
        self.rate = 0.05
        self.vol = 0.2
        self.strike = 50
        self.time = 0
        self.time_idx = 0
        self.expiry = 5
        self.expiry_idx = 2
        self.event_grid = np.array([self.time, self.expiry / 2, self.expiry])
        self.spot = np.arange(2, 200, 2)

    def test_decomposition(self) -> None:
        """Decompose call option price.

        The payoff of European put option can be decomposed into payoffs
        of European asset-or-nothing and cash-or-nothing put options
        written on same underlying:
            (K - S)^+ = K * I_{S < K} - S * I_{S < K}.
        """
        p = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.expiry_idx,
            self.event_grid, type_="Put")
        b_asset = binary.BinaryAssetPut(self.rate, self.vol, self.strike,
                                        self.expiry_idx, self.event_grid)
        b_cash = binary.BinaryCashPut(self.rate, self.vol, self.strike,
                                      self.expiry_idx, self.event_grid)
        price_p = p.price(self.spot, self.time_idx)
        price_ba = b_asset.price(self.spot, self.time_idx)
        price_bc = self.strike * b_cash.price(self.spot, self.time_idx)
        put_decomposed = - price_ba + price_bc
        diff = np.abs(price_p - put_decomposed)
        if print_results:
            print(np.max(diff))
        self.assertTrue(np.max(diff) < 1.0e-14)
        if plot_results:
            plt.plot(self.spot, p.payoff(self.spot), "-k", label="Put payoff")
            plt.plot(self.spot, price_p, "-ob", label="Put")
            plt.plot(self.spot, price_ba, "-r", label="Asset-or-nothing put")
            plt.plot(self.spot, price_bc, "-g", label="Cash-or-nothing put")
            plt.plot(self.spot, put_decomposed, "-y", label="Composition")
            plt.title("Put option, Black-Scholes model")
            plt.xlabel("Stock price")
            plt.ylabel("Option price")
            plt.legend()
            plt.pause(2)
            plt.clf()


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

        self.exercise_indices_mc1 = (
            np.arange(1 * self.frequency_mc, 0, -self.skip))
        self.exercise_indices_mc2 = (
            np.arange(2 * self.frequency_mc, 0, -self.skip))

        # Spatial grid used in FD pricing.
        self.x_grid = np.arange(801) / 4
        self.x_grid = self.x_grid[1:]

        # Number of MC paths.
        self.n_paths = 10000

        self.pFDa11 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_mc1,
            self.event_grid_fd1, type_="Put")

        self.pMCa11 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_mc1,
            self.event_grid_mc1, type_="Put")

        self.p11 = option.EuropeanOption(
            self.rate, self.vol1, self.strike, self.event_grid_fd1.size - 1,
            self.event_grid_fd1, type_="Put")

        self.pFDa12 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_mc2,
            self.event_grid_fd2, type_="Put")

        self.pMCa12 = option.AmericanOption(
            self.rate, self.vol1, self.strike, self.exercise_indices_mc2,
            self.event_grid_mc2, type_="Put")

        self.p12 = option.EuropeanOption(
            self.rate, self.vol1, self.strike, self.event_grid_fd2.size - 1,
            self.event_grid_fd2, type_="Put")

        self.pFDa21 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_mc1,
            self.event_grid_fd1, type_="Put")

        self.pMCa21 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_mc1,
            self.event_grid_mc1, type_="Put")

        self.p21 = option.EuropeanOption(
            self.rate, self.vol2, self.strike, self.event_grid_fd1.size - 1,
            self.event_grid_fd1, type_="Put")

        self.pFDa22 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_mc2,
            self.event_grid_fd2, type_="Put")

        self.pMCa22 = option.AmericanOption(
            self.rate, self.vol2, self.strike, self.exercise_indices_mc2,
            self.event_grid_mc2, type_="Put")

        self.p22 = option.EuropeanOption(
            self.rate, self.vol2, self.strike, self.event_grid_fd2.size - 1,
            self.event_grid_fd2, type_="Put")

    def test_pricing(self):
        """..."""
        self.pFDa11.fd_setup(self.x_grid, equidistant=True)
        self.pMCa11.mc_exact_setup_new()
        self.p11.mc_exact_setup()
        self.pFDa11.fd_solve()
        analytical11 = self.p11.price(self.x_grid, 0)

        self.pFDa12.fd_setup(self.x_grid, equidistant=True)
        self.pMCa12.mc_exact_setup_new()
        self.p12.mc_exact_setup()
        self.pFDa12.fd_solve()
        analytical12 = self.p12.price(self.x_grid, 0)

        self.pFDa21.fd_setup(self.x_grid, equidistant=True)
        self.pMCa21.mc_exact_setup_new()
        self.p21.mc_exact_setup()
        self.pFDa21.fd_solve()
        analytical21 = self.p21.price(self.x_grid, 0)

        self.pFDa22.fd_setup(self.x_grid, equidistant=True)
        self.pMCa22.mc_exact_setup_new()
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

            self.pMCa11.mc_exact_solve_new(
                y, self.n_paths, seed=0, antithetic=True)
            pa11_mc = lsm.american_option(self.pMCa11)

            self.p12.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p12_mean = self.p12.mc_exact.mc_estimate
            p12_error = self.p12.mc_exact.mc_error

            self.pMCa12.mc_exact_solve_new(
                y, self.n_paths, seed=0, antithetic=True)
            pa12_mc = lsm.american_option(self.pMCa12)

            self.p21.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p21_mean = self.p21.mc_exact.mc_estimate
            p21_error = self.p21.mc_exact.mc_error

            self.pMCa21.mc_exact_solve_new(
                y, self.n_paths, seed=0, antithetic=True)
            pa21_mc = lsm.american_option(self.pMCa21)

            self.p22.mc_exact_solve(
                y, self.n_paths, seed=0, antithetic=True)
            p22_mean = self.p22.mc_exact.mc_estimate
            p22_error = self.p22.mc_exact.mc_error

            self.pMCa22.mc_exact_solve_new(
                y, self.n_paths, seed=0, antithetic=True)
            pa22_mc = lsm.american_option(self.pMCa22)

            for x, pa, p in \
                    zip(self.x_grid, self.pFDa11.fd.solution, analytical11):
                if abs(x - y) < 1.e-6:
                    diff = self.fd_american[counter] - pa
                    counter += 1
                    self.assertTrue(abs(diff) < self.threshold)
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
                    self.assertTrue(abs(diff) < self.threshold)
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
                    self.assertTrue(abs(diff) < self.threshold)
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
                    self.assertTrue(abs(diff) < self.threshold)
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

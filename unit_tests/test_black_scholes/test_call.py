import unittest

from matplotlib import pyplot as plt
import numpy as np

from numerics.mc import lsm
from models.black_scholes import call_option as call
from models.black_scholes import put_option as put
from models.black_scholes import binary_option as binary
from numerics.fd import grid_generation as grid
from utils import plots

plot_results = False
print_results = True


class CallOption(unittest.TestCase):
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

    def test_expiry(self) -> None:
        """Test expiry property."""
        c = call.Call(self.rate, self.vol, self.strike, self.expiry_idx,
                      self.event_grid)
        self.assertTrue(c.expiry == self.expiry)

    def test_decomposition(self) -> None:
        """Decompose call option price.

        The payoff of European call option can be decomposed into
        payoffs of European asset-or-nothing and cash-or-nothing call
        options written on same underlying:
            (S - K)^+ = S * I_{S > K} - K * I_{S > K}.
        """
        c = call.Call(self.rate, self.vol, self.strike, self.expiry_idx,
                      self.event_grid)
        b_asset = binary.BinaryAssetCall(self.rate, self.vol, self.strike,
                                         self.expiry_idx, self.event_grid)
        b_cash = binary.BinaryCashCall(self.rate, self.vol, self.strike,
                                       self.expiry_idx, self.event_grid)
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

    def test_greeks_by_fd(self) -> None:
        """Finite difference approximation of greeks."""
        n_steps = 500
        dt = (self.event_grid[-1] - self.event_grid[0]) / (n_steps - 1)
        event_grid = dt * np.arange(n_steps) + self.event_grid[0]
        c = call.Call(self.rate, self.vol, self.strike, event_grid.size - 1,
                      event_grid)

        equidistant = False
        x_steps = 100
        if equidistant:
            # Equidistant grid.
            x_grid = grid.equidistant(self.spot[0], self.spot[-1], x_steps)
        else:
            # Non-equidistant grid.
            _, x_grid = grid.hyperbolic(self.spot[0], self.spot[-1],
                                        x_steps, self.strike)

        c.fd_setup(x_grid, equidistant=equidistant)

        c.fd.solution = c.payoff(x_grid)
        c.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(c)
        # Check convergence in reduced interval around strike.
        idx_min = np.argwhere(x_grid < self.strike - 25)[-1][0]
        idx_max = np.argwhere(x_grid < self.strike + 25)[-1][0]
        # Compare delta.
        diff = (c.delta(x_grid, 0) - c.fd.delta()) / c.delta(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-3)
        # Compare gamma.
        diff = (c.gamma(x_grid, 0) - c.fd.gamma()) / c.gamma(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 3.0e-3)
        # Compare theta.
        diff = (c.theta(x_grid, 0) - c.fd.theta(0.0001)) / c.theta(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 1.0e-3)
        # Compare rho.
        new_rate = self.rate * 1.0001
        c_rho = call.Call(new_rate, self.vol, self.strike, event_grid.size - 1,
                          event_grid)
        c_rho.fd_setup(x_grid, equidistant=equidistant)
        c_rho.fd.solution = c.payoff(x_grid)
        c_rho.fd_solve()
        rho = (c_rho.fd.solution - c.fd.solution) / (new_rate - self.rate)
        if plot_results:
            plt.plot(x_grid, rho, "-b")
            plt.plot(x_grid, c.rho(x_grid, 0), "-r")
            plt.xlabel("Stock price")
            plt.ylabel("Rho")
            plt.pause(2)
            plt.clf()
        diff = (c.rho(x_grid, 0) - rho) / c.rho(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-3)
        # Compare vega.
        new_vol = self.vol * 1.00001
        c_vega = call.Call(self.rate, new_vol, self.strike,
                           event_grid.size - 1, event_grid)
        c_vega.fd_setup(x_grid, equidistant=equidistant)
        c_vega.fd.solution = c.payoff(x_grid)
        c_vega.fd_solve()
        vega = (c_vega.fd.solution - c.fd.solution) / (new_vol - self.vol)
        if plot_results:
            plt.plot(x_grid, vega, "-b")
            plt.plot(x_grid, c.vega(x_grid, 0), "-r")
            plt.xlabel("Stock price")
            plt.ylabel("Vega")
            plt.pause(2)
            plt.clf()
        diff = (c.vega(x_grid, 0) - vega) / c.vega(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 3.0e-3)

    def test_monte_carlo(self) -> None:
        """Monte-Carlo simulation."""
        t_steps = 100
        expiry_idx = t_steps - 1
        dt = (self.expiry - self.time) / (t_steps - 1)
        integration_grid = dt * np.arange(t_steps) + self.time
        c = call.Call(self.rate, self.vol, self.strike, expiry_idx,
                      integration_grid)
        c.mc_exact_setup()
        n_paths = 100
        mc_spot = np.arange(25, 200, 25)
        price_array = np.zeros(mc_spot.size)
        std_array = np.zeros(mc_spot.size)
        mc_error = np.zeros(mc_spot.size)
        for idx, s in enumerate(mc_spot):
            c.mc_exact.initialization(s, n_paths, antithetic=True)
            c.mc_exact_solve()
            if plot_results:
                plt.plot(c.event_grid, c.mc_exact.solution)
                plt.xlabel("Time")
                plt.ylabel("Stock price")
                plt.pause(1)
                plt.clf()
            price_array[idx], std_array[idx], mc_error[idx] = \
                c.mc_exact.price(c, expiry_idx)
        if plot_results:
            plt.plot(self.spot, c.payoff(self.spot), "-k")
            plt.plot(self.spot, c.price(self.spot, 0), "-r")
            plt.errorbar(mc_spot, price_array, yerr=mc_error,
                         linestyle="none", marker="o", color="b", capsize=5)
            plt.xlabel("Stock price")
            plt.ylabel("Option price")
            plt.pause(5)


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
            call.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_fd1)

        self.pMCa11 = \
            call.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_mc1)

        self.p11 = \
            call.Call(self.rate,
                    self.vol1,
                    self.strike,
                    self.event_grid_fd1.size - 1,
                    self.event_grid_fd1)

        self.pFDa12 = \
            call.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_fd2)

        self.pMCa12 = \
            call.CallAmerican(self.rate,
                            self.vol1,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_mc2)

        self.p12 = \
            call.Call(self.rate,
                    self.vol1,
                    self.strike,
                    self.event_grid_fd2.size - 1,
                    self.event_grid_fd2)

        self.pFDa21 = \
            call.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_fd1)

        self.pMCa21 = \
            call.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc1,
                            self.event_grid_mc1)

        self.p21 = \
            call.Call(self.rate,
                    self.vol2,
                    self.strike,
                    self.event_grid_fd1.size - 1,
                    self.event_grid_fd1)

        self.pFDa22 = \
            call.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_fd2)

        self.pMCa22 = \
            call.CallAmerican(self.rate,
                            self.vol2,
                            self.strike,
                            self.exercise_indices_mc2,
                            self.event_grid_mc2)

        self.p22 = \
            call.Call(self.rate,
                    self.vol2,
                    self.strike,
                    self.event_grid_fd2.size - 1,
                    self.event_grid_fd2)

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

            self.p11.mc_exact.initialization(y, self.n_paths,
                                             seed=0, antithetic=True)
            self.p11.mc_exact_solve()
            p11_mean, _, p11_error = \
                self.p11.mc_exact.price(self.p11, self.event_grid_fd1.size - 1)

            self.pMCa11.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa11.mc_exact_solve()
            pa11_mc = lsm.american_option(self.pMCa11)

            self.p12.mc_exact.initialization(y, self.n_paths,
                                             seed=0, antithetic=True)
            self.p12.mc_exact_solve()
            p12_mean, _, p12_error = \
                self.p12.mc_exact.price(self.p12, self.event_grid_fd2.size - 1)

            self.pMCa12.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa12.mc_exact_solve()
            pa12_mc = lsm.american_option(self.pMCa12)

            self.p21.mc_exact.initialization(y, self.n_paths,
                                             seed=0, antithetic=True)
            self.p21.mc_exact_solve()
            p21_mean, _, p21_error = \
                self.p21.mc_exact.price(self.p21, self.event_grid_fd1.size - 1)

            self.pMCa21.mc_exact.initialization(y, self.n_paths,
                                                seed=0, antithetic=True)
            self.pMCa21.mc_exact_solve()
            pa21_mc = lsm.american_option(self.pMCa21)

            self.p22.mc_exact.initialization(y, self.n_paths,
                                             seed=0, antithetic=True)
            self.p22.mc_exact_solve()
            p22_mean, _, p22_error = \
                self.p22.mc_exact.price(self.p22, self.event_grid_fd2.size - 1)

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


if __name__ == '__main__':
    unittest.main()

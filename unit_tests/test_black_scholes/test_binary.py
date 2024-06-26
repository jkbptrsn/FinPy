import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import european_option as option
from models.black_scholes import binary_option as binary
from utils import plots

plot_results = False
print_results = False

if print_results:
    print("Unit test results from: " + __name__)


class Decomposition(unittest.TestCase):

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

    def test_call(self) -> None:
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

    def test_decomposition(self) -> None:
        """Decompose put option price.

        The payoff of European put option can be decomposed into payoffs
        of European asset-or-nothing and cash-or-nothing put options
        written on same underlying:
            (K - S)^+ = K * I_{S < K} - S * I_{S < K}.
        """
        p = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid,
            type_="Put")
        b_asset = binary.BinaryAssetPut(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid)
        b_cash = binary.BinaryCashPut(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid)
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
            plt.show()


class BinaryCashCall(unittest.TestCase):
    """European cash-or-nothing call option in Black-Scholes model."""

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

    def test_greeks_by_fd(self) -> None:
        """Finite difference approximation of greeks."""
        n_steps = 500
        dt = (self.event_grid[-1] - self.event_grid[0]) / (n_steps - 1)
        event_grid = dt * np.arange(n_steps) + self.event_grid[0]
        b = binary.BinaryCashCall(self.rate, self.vol, self.strike,
                                  event_grid.size - 1, event_grid)
        x_steps = 500
        dx = (self.spot[-1] - self.spot[0]) / (x_steps - 1)
        x_grid = dx * np.arange(x_steps) + self.spot[0]
        b.fd_setup(x_grid, equidistant=True)
        b.fd.solution = b.payoff(x_grid)
        b.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(b)
        # Check convergence in reduced interval around strike.
        idx_min = np.argwhere(x_grid < self.strike - 25)[-1][0]
        idx_max = np.argwhere(x_grid < self.strike + 25)[-1][0]
        # Compare delta.
        diff = b.delta(x_grid, 0) - b.fd.delta()
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 1.0e-4)
        # Compare gamma.
        diff = b.gamma(x_grid, 0) - b.fd.gamma()
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 1.0e-5)
        # Compare theta.
        diff = b.theta(x_grid, 0) - b.fd.theta(0.0001)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 3.0e-4)
        # Compare rho.
        new_rate = self.rate * 1.0001
        b_rho = binary.BinaryCashCall(new_rate, self.vol, self.strike,
                                      event_grid.size - 1, event_grid)
        b_rho.fd_setup(x_grid, equidistant=True)
        b_rho.fd.solution = b.payoff(x_grid)
        b_rho.fd_solve()
        rho = (b_rho.fd.solution - b.fd.solution) / (new_rate - self.rate)
        if plot_results:
            plt.plot(x_grid, rho, "-b")
            plt.plot(x_grid, b.rho(x_grid, 0), "-r")
            plt.xlabel("Stock price")
            plt.ylabel("Rho")
            plt.pause(2)
            plt.clf()
        diff = b.rho(x_grid, 0) - rho
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-2)
        # Compare vega.
        new_vol = self.vol * 1.00001
        b_vega = binary.BinaryCashCall(self.rate, new_vol, self.strike,
                                       event_grid.size - 1, event_grid)
        b_vega.fd_setup(x_grid, equidistant=True)
        b_vega.fd.solution = b.payoff(x_grid)
        b_vega.fd_solve()
        vega = (b_vega.fd.solution - b.fd.solution) / (new_vol - self.vol)
        if plot_results:
            plt.plot(x_grid, vega, "-b")
            plt.plot(x_grid, b.vega(x_grid, 0), "-r")
            plt.xlabel("Stock price")
            plt.ylabel("Vega")
            plt.pause(2)
            plt.clf()
        diff = b.vega(x_grid, 0) - vega
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 1.0e-2)

import math
import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import call_option as call
from models.black_scholes import binary_option as binary
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
        x_steps = 200
        if equidistant:
            # Equidistant grid.
            dx = (self.spot[-1] - self.spot[0]) / (x_steps - 1)
            x_grid = dx * np.arange(x_steps) + self.spot[0]
        else:
            # Non-equidistant grid. Hout & Foulon, 2010.
            const_c = 10
            d_eps = (math.asinh((self.spot[-1] - self.strike) / const_c)
                     - math.asinh(-self.strike / const_c)) / x_steps
            eps_grid = d_eps * np.arange(x_steps) \
                + math.asinh(-self.strike / const_c)
            x_grid = self.strike + const_c * np.sinh(eps_grid)
            x_grid = x_grid[1:]
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
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-4)
        # Compare gamma.
        diff = (c.gamma(x_grid, 0) - c.fd.gamma()) / c.gamma(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 4.0e-4)
        # Compare theta.
        diff = (c.theta(x_grid, 0) - c.fd.theta(0.0001)) / c.theta(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-4)
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
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 3.0e-4)
        # Compare vega.
        new_vol = self.vol * 1.00001
        c_vega = call.Call(self.rate, new_vol, self.strike, event_grid.size - 1, event_grid)
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
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 5.0e-4)

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


if __name__ == '__main__':
    unittest.main()

import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import put
from models.black_scholes import binary
from utils import plots

plot_results = False
print_results = False


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

    def test_expiry(self) -> None:
        """Test expiry property."""
        p = put.Put(self.rate, self.vol, self.strike, self.expiry_idx,
                    self.event_grid)
        self.assertTrue(p.expiry == self.expiry)

    def test_decomposition(self) -> None:
        """Decompose call option price.

        The payoff of European put option can be decomposed into payoffs
        of European asset-or-nothing and cash-or-nothing put options
        written on same underlying:
            (K - S)^+ = K * I_{S < K} - S * I_{S < K}.
        """
        p = put.Put(self.rate, self.vol, self.strike, self.expiry_idx,
                    self.event_grid)
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

    def test_greeks_by_fd(self) -> None:
        """Finite difference approximation of greeks."""
        n_steps = 500
        dt = (self.event_grid[-1] - self.event_grid[0]) / (n_steps - 1)
        event_grid = dt * np.arange(n_steps) + self.event_grid[0]
        p = put.Put(self.rate, self.vol, self.strike, event_grid.size - 1,
                    event_grid)
        x_steps = 500
        dx = (self.spot[-1] - self.spot[0]) / (x_steps - 1)
        x_grid = dx * np.arange(x_steps) + self.spot[0]
        p.fd_setup(x_grid, equidistant=True)
        p.fd.solution = p.payoff(x_grid)
        p.fd_solve()
        if plot_results:
            plots.plot_price_and_greeks(p, p.payoff(x_grid), p.price(x_grid, 0))
        # Check convergence in reduced interval around strike.
        idx_min = np.argwhere(x_grid < self.strike - 25)[-1][0]
        idx_max = np.argwhere(x_grid < self.strike + 25)[-1][0]
        # Compare delta.
        diff = (p.delta(x_grid, 0) - p.fd.delta()) / p.delta(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 8.0e-2)
        # Compare gamma.
        diff = (p.gamma(x_grid, 0) - p.fd.gamma()) / p.gamma(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-2)
        # Compare theta. Use absolute difference...
        diff = (p.theta(x_grid, 0) - p.fd.theta(0.001))
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-4)
        # Compare rho.
        new_rate = self.rate * 1.0001
        p_rho = put.Put(new_rate, self.vol, self.strike, event_grid.size - 1,
                        event_grid)
        p_rho.fd_setup(x_grid, equidistant=True)
        p_rho.fd.solution = p.payoff(x_grid)
        p_rho.fd_solve()
        rho = (p_rho.fd.solution - p.fd.solution) / (new_rate - self.rate)
        if plot_results:
            plt.plot(x_grid, rho, "-b")
            plt.plot(x_grid, p.rho(x_grid, 0), "-r")
            plt.xlabel("Stock price")
            plt.ylabel("Rho")
            plt.pause(2)
            plt.clf()
        diff = (p.rho(x_grid, 0) - rho) / p.rho(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-5)
        # Compare vega.
        new_vol = self.vol * 1.00001
        p_vega = put.Put(self.rate, new_vol, self.strike, event_grid.size - 1, event_grid)
        p_vega.fd_setup(x_grid, equidistant=True)
        p_vega.fd.solution = p.payoff(x_grid)
        p_vega.fd_solve()
        vega = (p_vega.fd.solution - p.fd.solution) / (new_vol - self.vol)
        if plot_results:
            plt.plot(x_grid, vega, "-b")
            plt.plot(x_grid, p.vega(x_grid, 0), "-r")
            plt.xlabel("Stock price")
            plt.ylabel("Vega")
            plt.pause(2)
            plt.clf()
        diff = (p.vega(x_grid, 0) - vega) / p.vega(x_grid, 0)
        if print_results:
            print(np.max(np.abs(diff[idx_min:idx_max])))
        self.assertTrue(np.max(np.abs(diff[idx_min:idx_max])) < 2.0e-4)


if __name__ == '__main__':
    unittest.main()

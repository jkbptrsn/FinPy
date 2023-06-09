import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import put_option as put
from models.black_scholes import binary_option as binary
from utils import plots

plot_results = True
print_results = True


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
            plots.plot_price_and_greeks(p)
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


class LongstaffSchwartz(unittest.TestCase):
    """Numerical examples in Longstaff-Schwartz article."""

    def setUp(self) -> None:
        self.rate = 0.06
        self.vol1 = 0.2
        self.vol2 = 0.4
        self.strike = 40

        self.frequency = 50
        self.event_grid1 = \
            np.arange(int(1 * self.frequency) + 1) / self.frequency
        self.event_grid2 = \
            np.arange(int(2 * self.frequency) + 1) / self.frequency

        self.x_grid = np.arange(1, 4802) / 24

        self.pa11 = put.PutAmerican(self.rate,
                                    self.vol1,
                                    self.strike,
                                    self.event_grid1.size - 1,
                                    self.event_grid1)

        self.p11 = put.Put(self.rate,
                           self.vol1,
                           self.strike,
                           self.event_grid1.size - 1,
                           self.event_grid1)

        self.pa12 = put.PutAmerican(self.rate,
                                    self.vol1,
                                    self.strike,
                                    self.event_grid2.size - 1,
                                    self.event_grid2)

        self.p12 = put.Put(self.rate,
                           self.vol1,
                           self.strike,
                           self.event_grid2.size - 1,
                           self.event_grid2)

        self.pa21 = put.PutAmerican(self.rate,
                                    self.vol2,
                                    self.strike,
                                    self.event_grid1.size - 1,
                                    self.event_grid1)

        self.p21 = put.Put(self.rate,
                           self.vol2,
                           self.strike,
                           self.event_grid1.size - 1,
                           self.event_grid1)

        self.pa22 = put.PutAmerican(self.rate,
                                    self.vol2,
                                    self.strike,
                                    self.event_grid2.size - 1,
                                    self.event_grid2)

        self.p22 = put.Put(self.rate,
                           self.vol2,
                           self.strike,
                           self.event_grid2.size - 1,
                           self.event_grid2)

    def test_pricing(self):
        """..."""
        self.pa11.fd_setup(self.x_grid, equidistant=True)
        self.pa11.fd_solve()

        n_paths = 100000
        self.p11.mc_exact_setup()
        self.p11.mc_exact.initialization(36, n_paths, seed=0, antithetic=True)
        self.p11.mc_exact_solve()
        mean, std, error = self.p11.mc_exact.price(self.p11, self.event_grid1.size - 1)
        print(mean, std, error)
        analytical11 = self.p11.price(self.x_grid, 0)

        self.pa12.fd_setup(self.x_grid, equidistant=True)
        self.pa12.fd_solve()
        analytical12 = self.p12.price(self.x_grid, 0)

        self.pa21.fd_setup(self.x_grid, equidistant=True)
        self.pa21.fd_solve()
        analytical21 = self.p21.price(self.x_grid, 0)

        self.pa22.fd_setup(self.x_grid, equidistant=True)
        self.pa22.fd_solve()
        analytical22 = self.p22.price(self.x_grid, 0)
        if print_results:
            for y in (36, 38, 40, 42, 44):
                for x, pa, p in \
                        zip(self.x_grid, self.pa11.fd.solution, analytical11):
                    if abs(x - y) < 1.e-6:
                        print(f"{int(x):3}  "
                              f"{pa:6.3f}  "
                              f"{p:6.3f}  "
                              f"{pa - p:6.3f}")
                for x, pa, p in \
                        zip(self.x_grid, self.pa12.fd.solution, analytical12):
                    if abs(x - y) < 1.e-6:
                        print(f"{int(x):3}  "
                              f"{pa:6.3f}  "
                              f"{p:6.3f}  "
                              f"{pa - p:6.3f}")
                for x, pa, p in \
                        zip(self.x_grid, self.pa21.fd.solution, analytical21):
                    if abs(x - y) < 1.e-6:
                        print(f"{int(x):3}  "
                              f"{pa:6.3f}  "
                              f"{p:6.3f}  "
                              f"{pa - p:6.3f}")
                for x, pa, p in \
                        zip(self.x_grid, self.pa22.fd.solution, analytical22):
                    if abs(x - y) < 1.e-6:
                        print(f"{int(x):3}  "
                              f"{pa:6.3f}  "
                              f"{p:6.3f}  "
                              f"{pa - p:6.3f}")
                print("")
        if plot_results:
            plt.plot(self.x_grid, self.pa11.fd.solution)
            plt.show()


if __name__ == '__main__':
    unittest.main()

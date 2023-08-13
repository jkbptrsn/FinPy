import unittest

import matplotlib.pyplot as plt
import numpy as np

from models.hull_white import callable_bond as bond
from unit_tests.test_hull_white import input
from utils import cash_flows
from utils import plots

plot_results = False
print_results = False


class FixedRate(unittest.TestCase):
    """Fixed rate callable bond in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        # Cash flow type.
        self.type = "deferred"
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
        self.io = 12
        # Number of terms in issuance period.
        self.issuance_terms = 4

        # Cash flow with issuance period.
        self.cash_flow_grid = \
            cash_flows.set_cash_flow_grid_issuance(
                self.t_i, self.t_f, self.frequency, self.issuance_terms)
        self.cash_flow = \
            cash_flows.cash_flow_split_issuance(
                self.coupon, self.frequency, self.cash_flow_grid,
                self.issuance_terms, self.principal, self.type, self.io)
#        self.cash_flow = self.cash_flow.sum(axis=0)

        # Move cash flow grid to "positive" times.
        self.cash_flow_grid += self.issuance_terms / self.frequency

        # Event grid.
        event_dt = 1 / 12
        self.event_grid, self.cash_flow_schedule, self.deadline_schedule = \
            cash_flows.set_event_grid(self.cash_flow_grid, event_dt)

        print(self.event_grid)
        print(self.cash_flow_schedule)
        print(self.deadline_schedule)

#        plt.plot(self.cash_flow_grid, self.cash_flow[0], "-b")
#        plt.plot(self.cash_flow_grid, self.cash_flow[1], "-r")
#        plt.show()

        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min
        self.time_dependence = "piecewise"
        # Bond.
        self.bond = \
            bond.FixedRate(self.kappa,
                           self.vol,
                           self.discount_curve,
                           self.coupon,
                           self.frequency,
                           self.deadline_schedule,
                           self.cash_flow_schedule,
                           self.cash_flow,
                           self.event_grid,
                           self.time_dependence)
        self.bond_pelsser = \
            bond.FixedRatePelsser(self.kappa,
                                  self.vol,
                                  self.discount_curve,
                                  self.coupon,
                                  self.frequency,
                                  self.deadline_schedule,
                                  self.cash_flow_schedule,
                                  self.cash_flow,
                                  self.event_grid,
                                  self.time_dependence)

    def test_theta_method(self):
        """Finite difference pricing of bond."""
        if print_results:
            print(self.bond.transformation)
        self.bond.fd_setup(self.x_grid, equidistant=True)
        self.bond.fd_solve()
        # Check price.
        numerical = self.bond.fd.solution
        analytical = self.bond.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.bond)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
#        self.assertTrue(max_error < 3.1e-5)
        # Check delta.
        numerical = self.bond.fd.delta()
        analytical = self.bond.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
#        self.assertTrue(max_error < 4.5e-5)
        # Check gamma.
        numerical = self.bond.fd.gamma()
        analytical = self.bond.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
#        self.assertTrue(max_error < 6.4e-5)
        # Check theta.
        numerical = self.bond.fd.theta()
        analytical = self.bond.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
#        self.assertTrue(max_error < 6.8e-4)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of bond."""
        if print_results:
            print(self.bond_pelsser.transformation)
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.bond_pelsser.fd_solve()
        # Check price.
        numerical = self.bond_pelsser.fd.solution
        analytical = self.bond_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.bond_pelsser)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
#        self.assertTrue(max_error < 2.3e-6)
        # Check delta.
        numerical = self.bond_pelsser.fd.delta()
        analytical = self.bond_pelsser.delta(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
#        self.assertTrue(max_error < 2.0e-6)
        # Check gamma.
        numerical = self.bond_pelsser.fd.gamma()
        analytical = self.bond_pelsser.gamma(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
#        self.assertTrue(max_error < 1.5e-6)
        # Check theta.
        numerical = self.bond_pelsser.fd.theta()
        analytical = self.bond_pelsser.theta(self.x_grid, 0)
        error = np.abs((analytical - numerical))
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
#        self.assertTrue(max_error < 7.2e-4)

    def test_theta_method_compare(self):
        """Finite difference pricing of bond."""
        if print_results:
            print(self.bond.transformation)
            print(self.bond_pelsser.transformation)
        self.bond.fd_setup(self.x_grid, equidistant=True)
        self.bond.fd_solve()
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.bond_pelsser.fd_solve()
        # Check price.
        numerical1 = self.bond.fd.solution
        numerical2 = self.bond_pelsser.fd.solution
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        if plot_results:
            plots.plot_price_and_greeks(self.bond)
        # Maximum error.
        idx_min = np.argwhere(self.x_grid < -0.02)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.02)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of price: {max_error:2.7f}")
#        self.assertTrue(max_error < 2.3e-6)
        # Check delta.
        numerical1 = self.bond.fd.delta()
        numerical2 = self.bond_pelsser.fd.delta()
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of delta: {max_error:2.7f}")
#        self.assertTrue(max_error < 2.0e-6)
        # Check gamma.
        numerical1 = self.bond.fd.gamma()
        numerical2 = self.bond_pelsser.fd.gamma()
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of gamma: {max_error:2.7f}")
        #        self.assertTrue(max_error < 1.5e-6)
        # Check theta.
        numerical1 = self.bond.fd.theta()
        numerical2 = self.bond_pelsser.fd.theta()
        relative_error = np.abs((numerical1 - numerical2) / numerical1)
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print(f"Maximum error of theta: {max_error:2.7f}")
#        self.assertTrue(max_error < 7.2e-4)

    def test_monte_carlo(self):
        """Monte-Carlo pricing of bond."""
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.bond.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.bond.mc_exact.mc_estimate
            error_exact[idx] = self.bond.mc_exact.mc_error
            self.bond.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.bond.mc_euler.mc_estimate
            error_euler[idx] = self.bond.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
#        self.assertTrue(max_error < 8.4e-3)

    def test_monte_carlo_compare(self):
        """Monte-Carlo pricing of bond."""
        self.bond.mc_exact_setup()
        self.bond.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000

        # FD result.
        self.bond.fd_setup(self.x_grid, equidistant=True)
        self.bond.fd_solve()
        price_a = self.bond.fd.solution

        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond.mc_exact_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_exact[idx] = self.bond.mc_exact.mc_estimate
            error_exact[idx] = self.bond.mc_exact.mc_error
            self.bond.mc_euler_solve(s, n_paths, rng=rng, antithetic=True)
            numerical_euler[idx] = self.bond.mc_euler.mc_estimate
            error_euler[idx] = self.bond.mc_euler.mc_error
        if plot_results:
            plt.plot(self.x_grid, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
#        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
#        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
#        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
#        max_error = np.max(error[idx_min:idx_max + 1])
#        if print_results:
#            print("max error: ", max_error)
#        self.assertTrue(max_error < 8.4e-3)

    def test_monte_carlo_pelsser(self):
        """Monte-Carlo pricing of bond."""
        self.bond_pelsser.mc_exact_setup()
        self.bond_pelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000
        # Analytical result.
        price_a = self.bond_pelsser.price(spot_vector, 0)
        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond_pelsser.mc_exact_solve(s, n_paths, rng=rng,
                                             antithetic=True)
            numerical_exact[idx] = self.bond_pelsser.mc_exact.mc_estimate
            error_exact[idx] = self.bond_pelsser.mc_exact.mc_error
            self.bond_pelsser.mc_euler_solve(s, n_paths, rng=rng,
                                             antithetic=True)
            numerical_euler[idx] = self.bond_pelsser.mc_euler.mc_estimate
            error_euler[idx] = self.bond_pelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(spot_vector, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
        max_error = np.max(error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
#        self.assertTrue(max_error < 8.4e-3)

    def test_monte_carlo_pelsser_compare(self):
        """Monte-Carlo pricing of bond."""
        self.bond_pelsser.mc_exact_setup()
        self.bond_pelsser.mc_euler_setup()
        # Spot rate.
        spot = 0.02
        spot_vector = spot * np.arange(11) - 0.1
        # Initialize random number generator.
        rng = np.random.default_rng(0)
        # Number of paths for each Monte-Carlo estimate.
        n_paths = 10000

        # FD result.
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.bond_pelsser.fd_solve()
        price_a = self.bond_pelsser.fd.solution

        numerical_exact = np.zeros(spot_vector.size)
        error_exact = np.zeros(spot_vector.size)
        numerical_euler = np.zeros(spot_vector.size)
        error_euler = np.zeros(spot_vector.size)
        for idx, s in enumerate(spot_vector):
            self.bond_pelsser.mc_exact_solve(s, n_paths, rng=rng,
                                             antithetic=True)
            numerical_exact[idx] = self.bond_pelsser.mc_exact.mc_estimate
            error_exact[idx] = self.bond_pelsser.mc_exact.mc_error
            self.bond_pelsser.mc_euler_solve(s, n_paths, rng=rng,
                                             antithetic=True)
            numerical_euler[idx] = self.bond_pelsser.mc_euler.mc_estimate
            error_euler[idx] = self.bond_pelsser.mc_euler.mc_error
        if plot_results:
            plt.plot(self.x_grid, price_a, "-b")
            plt.errorbar(spot_vector, numerical_exact, yerr=error_exact,
                         fmt='or', markersize=2, capsize=5, label="Exact")
            plt.errorbar(spot_vector, numerical_euler, yerr=error_euler,
                         fmt='og', markersize=2, capsize=5, label="Euler")
            plt.xlabel("Initial pseudo short rate")
            plt.ylabel("Bond price")
            plt.legend()
            plt.show()
#        error = np.abs(price_a - numerical_exact)
        # Maximum error in interval around pseudo short rate of 0.
#        idx_min = np.argwhere(spot_vector < -0.05)[-1][0]
#        idx_max = np.argwhere(spot_vector < 0.05)[-1][0]
#        max_error = np.max(error[idx_min:idx_max + 1])
#        if print_results:
#            print("max error: ", max_error)
#        self.assertTrue(max_error < 8.4e-3)


if __name__ == '__main__':
    unittest.main()

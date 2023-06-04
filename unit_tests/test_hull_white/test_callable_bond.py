import unittest

import numpy as np

from models.hull_white import callable_bond as bond
from unit_tests.test_hull_white import input
from utils import cash_flows
from utils import plots

plot_results = True
print_results = True


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
        self.io = 40
        # Number of terms in issuance period.
        self.issuance_terms = 12

        # Cash flow with issuance period.
        self.cash_flow_grid = \
            cash_flows.set_cash_flow_grid_issuance(
                self.t_i, self.t_f, self.frequency, self.issuance_terms)
        self.cash_flow_grid -= self.cash_flow_grid[0]
        self.cash_flow = \
            cash_flows.cash_flow_split_issuance(
                self.coupon, self.frequency, self.cash_flow_grid,
                self.issuance_terms, self.principal, self.type, self.io)
        self.cash_flow = self.cash_flow.sum(axis=0)

        # Event grid.
        event_dt = 1 / 12
        self.event_grid, self.cash_flow_schedule = \
            cash_flows.set_event_grid(self.cash_flow_grid, event_dt)

        # FD spatial grid.
        self.x_min = -0.15
        self.x_max = 0.15
        self.x_steps = 201
        self.dx = (self.x_max - self.x_min) / (self.x_steps - 1)
        self.x_grid = self.dx * np.arange(self.x_steps) + self.x_min

        # Bond.
        self.time_dependence = "piecewise"
        self.bond = bond.FixedRate(self.kappa,
                                   self.vol,
                                   self.discount_curve,
                                   self.cash_flow_schedule,
                                   self.cash_flow,
                                   self.event_grid,
                                   self.time_dependence)
        self.bond_pelsser = \
            bond.FixedRatePelsser(self.kappa,
                                  self.vol,
                                  self.discount_curve,
                                  self.cash_flow_schedule,
                                  self.cash_flow,
                                  self.event_grid,
                                  self.time_dependence)

    def test_theta_method(self):
        """Finite difference pricing of bond."""
        self.bond.fd_setup(self.x_grid, equidistant=True)
        self.bond.fd_solve()
        numerical = self.bond.fd.solution
        analytical = self.bond.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.bond)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
            print("Price at zero = ", analytical[(self.x_steps - 1) // 2])
        self.assertTrue(max_error < 9.e-4)

    def test_theta_method_pelsser(self):
        """Finite difference pricing of bond."""
        self.bond_pelsser.fd_setup(self.x_grid, equidistant=True)
        self.bond_pelsser.fd_solve()
        numerical = self.bond_pelsser.fd.solution
        analytical = self.bond_pelsser.price(self.x_grid, 0)
        relative_error = np.abs((analytical - numerical) / analytical)
        if plot_results:
            plots.plot_price_and_greeks(self.bond_pelsser)
        # Maximum error in interval around pseudo short rate of 0.
        idx_min = np.argwhere(self.x_grid < -0.05)[-1][0]
        idx_max = np.argwhere(self.x_grid < 0.05)[-1][0]
        max_error = np.max(relative_error[idx_min:idx_max + 1])
        if print_results:
            print("max error: ", max_error)
            print("Price at zero = ", analytical[(self.x_steps - 1) // 2])
        self.assertTrue(max_error < 9.e-4)


if __name__ == '__main__':
    unittest.main()

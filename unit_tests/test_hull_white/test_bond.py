import unittest

import numpy as np

from models.hull_white import bond
from unit_tests.test_hull_white import input
from utils import cash_flows
from utils import plots

plot_results = False
print_results = False


class Bond(unittest.TestCase):
    """Bond in 1-factor Hull-White model."""

    def setUp(self) -> None:
        # Model parameters.
        self.kappa = input.kappa_strip
        self.vol = input.vol_strip
        self.discount_curve = input.disc_curve

        self.t_initial = 0
        self.t_final = 5
        self.principal = 100
        self.coupon = 0.05
        self.frequency = 1
        self.cf_type = "annuity"

        self.cash_flow_grid = \
            cash_flows.set_cash_flow_grid(self.t_initial, self.t_final,
                                          self.frequency)
        self.cash_flow = \
            cash_flows.cash_flow(self.coupon, self.frequency,
                                 self.cash_flow_grid, self.principal,
                                 self.cf_type)

        # Event grid
        event_dt = 0.01
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
        self.bond = bond.Bond(self.kappa,
                              self.vol,
                              self.discount_curve,
                              self.cash_flow_schedule,
                              self.cash_flow,
                              self.event_grid,
                              self.time_dependence)
        self.bond_pelsser = \
            bond.BondPelsser(self.kappa,
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
            cash_flows.print_cash_flow(self.cash_flow)
            print("max error: ", max_error)
            print("Price at zero = ", analytical[(self.x_steps - 1) // 2])
        self.assertTrue(max_error < 4.e-4)

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
            cash_flows.print_cash_flow(self.cash_flow)
            print("max error: ", max_error)
            print("Price at zero = ", analytical[(self.x_steps - 1) // 2])
        self.assertTrue(max_error < 4.e-4)


if __name__ == '__main__':
    unittest.main()

import unittest

from matplotlib import pyplot as plt
import numpy as np

from utils import cash_flows

plot_results = False
print_results = False


class CashFlows(unittest.TestCase):

    def setUp(self) -> None:
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
        self.n_io_terms = 40
        # Number of terms in issuance period.
        self.issuance_terms = 12
        # Cash flow.
        self.payment_grid = \
            cash_flows.set_payment_grid(self.t_i,
                                        self.t_f,
                                        self.frequency)
        self.cash_flow = \
            cash_flows.cash_flow_split(self.coupon,
                                       self.frequency,
                                       self.payment_grid,
                                       self.principal,
                                       self.type,
                                       self.n_io_terms)
        # Cash flow grid with issuance period.
        self.payment_grid_issuance = \
            cash_flows.set_payment_grid_issuance(self.t_i,
                                                 self.t_f,
                                                 self.frequency,
                                                 self.issuance_terms)
        self.cash_flow_issuance = \
            cash_flows.cash_flow_split_issuance(self.coupon,
                                                self.frequency,
                                                self.payment_grid_issuance,
                                                self.issuance_terms,
                                                self.principal,
                                                self.type,
                                                self.n_io_terms)

    def test_standard_loans(self):
        """Standard loans."""
        annuity = \
            cash_flows.annuity(self.coupon,
                               self.frequency,
                               self.payment_grid,
                               100)
        self.assertTrue(np.max(np.abs(np.diff(annuity.sum(axis=0)))) < 1.0e-12)
        standing_loan = \
            cash_flows.standing_loan(self.coupon,
                                     self.frequency,
                                     self.payment_grid,
                                     100)
        self.assertTrue(np.max(np.abs(np.diff(standing_loan[1]))) < 1.0e-12)
        self.assertTrue(abs(standing_loan[0, -1] - 100) < 1.0e-12)
        serial_loan = \
            cash_flows.serial_loan(self.coupon,
                                   self.frequency,
                                   self.payment_grid,
                                   100)
        self.assertTrue(np.max(np.abs(np.diff(serial_loan[0]))) < 1.0e-12)
        if print_results:
            print("Annuity loan:")
            cash_flows.print_cash_flow(annuity)
            print("Standing loan:")
            cash_flows.print_cash_flow(standing_loan)
            print("Serial loan:")
            cash_flows.print_cash_flow(serial_loan)
        if plot_results:
            # Annuity.
            plt.plot(self.payment_grid, annuity[0, :],
                     "ob", label="Redemption")
            plt.plot(self.payment_grid, annuity[1, :],
                     "or", label="Interest")
            plt.plot(self.payment_grid, annuity.sum(axis=0),
                     "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()
            # Standing loan.
            plt.plot(self.payment_grid, standing_loan[0, :],
                     "ob", label="Redemption")
            plt.plot(self.payment_grid, standing_loan[1, :],
                     "or", label="Interest")
            plt.plot(self.payment_grid, standing_loan.sum(axis=0),
                     "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()
            # Serial loan.
            plt.plot(self.payment_grid, serial_loan[0, :],
                     "ob", label="Redemption")
            plt.plot(self.payment_grid, serial_loan[1, :],
                     "or", label="Interest")
            plt.plot(self.payment_grid, serial_loan.sum(axis=0),
                     "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()

    def test_relative_redemptions(self):
        """Recalculate redemptions backwards in time."""
        cf_test = np.zeros(self.cash_flow.shape)
        for n in range(cf_test.shape[1] - 1, -1, -1):
            # Redemption rate of remaining principal.
            redemption_rate = \
                self.cash_flow[0, n] / self.cash_flow[0, n:].sum()
            cf_test[0, n] = redemption_rate * self.principal
            cf_test[1, n] = self.coupon * self.principal / self.frequency
            cf_test[:, n + 1:] *= (1 - redemption_rate)
        for n in range(cf_test.shape[1]):
            diff1 = abs(cf_test[0, n] - self.cash_flow[0, n])
            diff2 = abs(cf_test[1, n] - self.cash_flow[1, n])
            self.assertTrue(diff1 < 1.e-13)
            self.assertTrue(diff2 < 1.e-13)
            if print_results:
                print(f"Term{n + 1:5}:",
                      f"{cf_test[0, n]:10.5f}",
                      f"{self.cash_flow[0, n]:10.5f}",
                      f"{cf_test[1, n]:10.5f}",
                      f"{self.cash_flow[1, n]:10.5f}")
        if print_results:
            print("Deferred annuity:")
            cash_flows.print_cash_flow(self.cash_flow)
            print("Deferred annuity with issuance period:")
            cash_flows.print_cash_flow(self.cash_flow_issuance)
        if plot_results:
            # Deferred annuity.
            plt.plot(self.payment_grid, self.cash_flow[0, :],
                     "ob", label="Installment")
            plt.plot(self.payment_grid, self.cash_flow[1, :],
                     "or", label="Interest")
            plt.plot(self.payment_grid, self.cash_flow.sum(axis=0),
                     "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()
            # Deferred annuity with issuance period.
            plt.plot(self.payment_grid_issuance,
                     self.cash_flow_issuance[0, :],
                     "ob", label="Installment")
            plt.plot(self.payment_grid_issuance,
                     self.cash_flow_issuance[1, :],
                     "or", label="Interest")
            plt.plot(self.payment_grid_issuance,
                     self.cash_flow_issuance.sum(axis=0),
                     "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    unittest.main()

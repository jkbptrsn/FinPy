import unittest

from matplotlib import pyplot as plt
import numpy as np

from utils import cash_flows

plot_results = True
print_results = True


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
        self.io = 40
        # Number of terms in issuance period.
        self.issuance_terms = 12
        # Cash flow.
        self.payment_grid = \
            cash_flows.set_cash_flow_grid(self.t_i,
                                          self.t_f,
                                          self.frequency)
        self.cash_flow = \
            cash_flows.cash_flow_split(self.coupon,
                                       self.frequency,
                                       self.payment_grid,
                                       self.principal,
                                       self.type,
                                       self.io)
        # Cash flow grid with issuance period.
        self.payment_grid_issuance = \
            cash_flows.set_cash_flow_grid_issuance(self.t_i,
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
                                                self.io)

    def test_normalization(self):
        """Test normalization of installments."""
        self.assertAlmostEqual(self.cash_flow_issuance[0, :].sum(), 100, 20)

    def test_plots(self):
        """..."""
        if print_results:
            cash_flows.print_cash_flow(self.cash_flow)
            print("")
            cash_flows.print_cash_flow(self.cash_flow_issuance)
        if plot_results:
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

        # Calculating redemptions backwards.
        cf_test = np.zeros(self.cash_flow.shape)
        for n in range(cf_test.shape[1] - 1, -1, -1):
            # Redemption rate of sum of remaining installments.
            redemption_rate = \
                self.cash_flow[0, n] / self.cash_flow[0, n:].sum()
            cf_test[0, n] = redemption_rate
            cf_test[1, n] = self.coupon / self.frequency
            cf_test[:, n + 1:] *= (1 - redemption_rate)
        for n in range(cf_test.shape[1]):
            diff1 = abs(100 * cf_test[0, n] - self.cash_flow[0, n])
            diff2 = abs(100 * cf_test[1, n] - self.cash_flow[1, n])
            self.assertTrue(diff1 < 1.e-13)
            self.assertTrue(diff2 < 1.e-13)
            if print_results:
                print(n,
                      100 * cf_test[0, n], self.cash_flow[0, n],
                      100 * cf_test[1, n], self.cash_flow[1, n])


if __name__ == '__main__':
    unittest.main()

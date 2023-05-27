import unittest

from matplotlib import pyplot as plt

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
        self.io = 40
        # Number of terms in issuance period.
        self.issuance_terms = 12
        # Cash flow.
        self.payment_grid = \
            cash_flows.set_cash_flow_grid(self.t_i, self.t_f, self.frequency)
        self.cash_flow = \
            cash_flows.cash_flow_split(
                self.coupon, self.frequency, self.payment_grid, self.principal,
                self.type, self.io)
        # Cash flow grid with issuance period.
        self.payment_grid_issuance = \
            cash_flows.set_cash_flow_grid_issuance(
                self.t_i, self.t_f, self.frequency, self.issuance_terms)
        self.cash_flow_issuance = \
            cash_flows.cash_flow_split_issuance(
                self.coupon, self.frequency, self.payment_grid_issuance,
                self.issuance_terms, self.principal, self.type, self.io)

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

            plt.plot(self.payment_grid_issuance, self.cash_flow_issuance[0, :],
                     "ob", label="Installment")
            plt.plot(self.payment_grid_issuance, self.cash_flow_issuance[1, :],
                     "or", label="Interest")
            plt.plot(self.payment_grid_issuance, self.cash_flow_issuance.sum(axis=0),
                     "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    unittest.main()

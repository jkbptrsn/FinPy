import unittest

from matplotlib import pyplot as plt

from utils import cash_flows

plot_results = False
print_results = False


class CashFlows(unittest.TestCase):

    def setUp(self) -> None:
        # Principal at time zero.
        self.principal = 100
        # Fixed yearly coupon.
        self.coupon = 0.10
        # Term frequency per year.
        self.frequency = 2
        # Number of remaining terms.
        self.n_terms = 120

        self.t_i = 0
        self.t_f = 10

        self.type = "annuity"

        self.payment_grid = \
            cash_flows.set_cash_flow_grid(self.t_i, self.t_f, self.frequency)
        self.cash_flow = \
            cash_flows.cash_flow_split(self.coupon, self.frequency,
                                       self.payment_grid, self.principal,
                                       self.type)

    def test_this(self):
        if print_results:
            cash_flows.print_cash_flow(self.cash_flow)
        if plot_results:
            plt.plot(self.payment_grid, self.cash_flow[0, :], "ob", label="Installment")
            plt.plot(self.payment_grid, self.cash_flow[1, :], "or", label="Interest")
            plt.plot(self.payment_grid, self.cash_flow.sum(axis=0), "ok", label="Total")
            plt.xlabel("Time")
            plt.ylabel("Payment")
            plt.legend()
            plt.show()
        self.assertEqual(2, 2)


if __name__ == '__main__':
    unittest.main()

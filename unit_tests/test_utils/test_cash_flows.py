import unittest

from matplotlib import pyplot as plt

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
        self.t_f = 10
        # Principal at time zero.
        self.principal = 100
        # Fixed yearly coupon.
        self.coupon = 0.10
        # Term frequency per year.
        self.frequency = 2
        # Number of 'interest only' terms.
        self.io = 4
        # Cash flow grid.
        self.payment_grid = \
            cash_flows.set_cash_flow_grid(self.t_i, self.t_f, self.frequency)
        # Cash flow.
        self.cash_flow = \
            cash_flows.cash_flow_split(self.coupon, self.frequency,
                                       self.payment_grid, self.principal,
                                       self.type, self.io)

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

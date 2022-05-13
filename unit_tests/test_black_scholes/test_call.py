import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.black_scholes.call as call


class CallOption(unittest.TestCase):

    def test_call(self):
        c = call.Call(0.05, 0.2, 50, 2)
        self.assertTrue(c.price(50, 0) > 0)


if __name__ == '__main__':

    # Plot call option price
    spot = np.arange(2, 100, 2) * 1.0
    c1 = call.Call(0.05, 0.2, 50, 6)
    plt.plot(spot, c1.payoff(spot), '-k')
    plt.plot(spot, c1.price(spot, 0), '-ob')
    plt.show()

    unittest.main()

import math
import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.black_scholes.call as call
import models.black_scholes.put as put


class Parity(unittest.TestCase):

    def test_1(self):
        rate = 0.05
        vol = 0.2
        strike = 50
        expiry = 2
        time = 0
        c = call.Call(rate, vol, strike, expiry)
        p = put.Put(rate, vol, strike, expiry)
        spot = np.arange(2, 98, 2) * 1.0
        discount = math.exp(-rate * (expiry - time))
        lhs = c.price(spot, time) - p.price(spot, time)
        rhs = spot - strike * discount
        self.assertTrue(np.max(np.abs(lhs - rhs)) < 1.0e-10)


if __name__ == '__main__':

    # Plot call option price
    spot = np.arange(2, 100, 2) * 1.0
    c1 = call.Call(0.05, 0.2, 50, 6)
    plt.plot(spot, c1.payoff(spot), '-k')
    plt.plot(spot, c1.price(spot, 0), '-ob')
    plt.show()

    unittest.main()

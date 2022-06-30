import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.black_scholes.call as call
import models.black_scholes.binary as binary


class CallOption(unittest.TestCase):

    def test_binary_asset_and_cash_calls(self):
        # Parameters
        rate = 0.05
        vol = 0.2
        strike = 50
        expiry = 2
        time = 0
        spot = np.arange(2, 100, 2) * 1.0

        c1 = call.Call(rate, vol, strike, expiry)
        b1 = binary.BinaryAssetCall(rate, vol, strike, expiry)
        b2 = binary.BinaryCashCall(rate, vol, strike, expiry)
        diff = np.abs(c1.price(spot, time)
                      - (b1.price(spot, time) - strike * b2.price(spot, time)))
        self.assertTrue(np.max(diff) < 1.0e-12)


if __name__ == '__main__':

    # Plot call option price
    rate = 0.05
    vol = 0.2
    strike = 50
    expiry = 2
    expiry_idx = 1
    time = 0
    spot = np.arange(2, 100, 2) * 1.0
    event_grid = np.array([time, expiry])
    c1 = call.Call(rate, vol, event_grid, strike, expiry_idx)
#    b1 = binary.BinaryAssetCall(rate, vol, strike, expiry)
#    b2 = binary.BinaryCashCall(rate, vol, strike, expiry)
    plt.plot(spot, c1.payoff(spot), '-k')
    plt.plot(spot, c1.price(spot, 0), '-ob')
#    plt.plot(spot, b1.price(spot, 0), '-r')
#    plt.plot(spot, strike * b2.price(spot, 0), '-g')
#    plt.plot(spot, b1.price(spot, 0) - strike * b2.price(spot, 0), '-y')
    plt.show()

    unittest.main()

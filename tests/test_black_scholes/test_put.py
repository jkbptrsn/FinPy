import matplotlib.pyplot as plt
import numpy as np
import unittest

import models.black_scholes.put as put
import models.black_scholes.binary as binary


class PutOption(unittest.TestCase):

    def test_binary_asset_and_cash_puts(self):
        rate = 0.05
        vol = 0.2
        time = 0
        time_idx = 0
        expiry = 2
        expiry_idx = 1
        event_grid = np.array([time, expiry])
        strike = 50
        spot = np.arange(2, 100, 2) * 1.0
        p = put.Put(rate, vol, event_grid, strike, expiry_idx)
        b1 = binary.BinaryAssetPut(rate, vol, event_grid, strike, expiry_idx)
        b2 = binary.BinaryCashPut(rate, vol, event_grid, strike, expiry_idx)
        put_price_decomposed = \
            - b1.price(spot, time_idx) + strike * b2.price(spot, time_idx)
        diff = np.abs(p.price(spot, time) - put_price_decomposed)
        self.assertTrue(np.max(diff) < 1.0e-12)


if __name__ == '__main__':

    # Plot put option price and decompose into binary asset/cash put prices
    rate = 0.05
    vol = 0.2

    time = 0
    time_idx = 0
    expiry = 2
    expiry_idx = 1
    event_grid = np.array([time, expiry])

    strike = 50

    spot = np.arange(2, 100, 2) * 1.0

    p = put.Put(rate, vol, event_grid, strike, expiry_idx)
    b1 = binary.BinaryAssetPut(rate, vol, event_grid, strike, expiry_idx)
    b2 = binary.BinaryCashPut(rate, vol, event_grid, strike, expiry_idx)

    plt.plot(spot, p.payoff(spot), '-k')
    plt.plot(spot, p.price(spot, time_idx), '-ob')
    plt.plot(spot, b1.price(spot, time_idx), '-r')
    plt.plot(spot, strike * b2.price(spot, time_idx), '-g')
    put_price_decomposed = \
        - b1.price(spot, time_idx) + strike * b2.price(spot, time_idx)
    plt.plot(spot, put_price_decomposed, '-y')
    plt.show()

    unittest.main()

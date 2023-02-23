import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import put
from models.black_scholes import binary


class PutOption(unittest.TestCase):
    """Test European put option in Black-Scholes model."""

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

    plt.plot(spot, p.payoff(spot), "-k", label="Option payoff")
    plt.plot(spot, p.price(spot, time_idx), "-ob", label="Put")
    plt.plot(spot, b1.price(spot, time_idx), "-r", label="Binary asset put")
    plt.plot(spot, strike * b2.price(spot, time_idx), "-g", label="Binary cash put")
    put_price_decomposed = \
        - b1.price(spot, time_idx) + strike * b2.price(spot, time_idx)
    plt.plot(spot, put_price_decomposed, "-y", label="Combine binaries")
    plt.title("Put option, Black-Scholes model")
    plt.xlabel("Stock price")
    plt.ylabel("Put price")
    plt.legend()
    plt.show()

    unittest.main()

import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import call
from models.black_scholes import binary


class CallOption(unittest.TestCase):
    """European call options in Black-Scholes model."""

    def setUp(self) -> None:
        self.rate = 0.05
        self.vol = 0.2
        self.strike = 50
        self.time = 0
        self.time_idx = 0
        self.expiry = 5
        self.expiry_idx = 2
        self.event_grid = np.array([self.time, self.expiry / 2, self.expiry])
        self.spot = np.arange(1, 100)

    def test_expiry(self) -> None:
        """Test expiry property."""
        c = call.Call(self.rate, self.vol, self.strike, self.expiry_idx,
                      self.event_grid)
        self.assertTrue(c.expiry == self.expiry)

    def test_decompose(self) -> None:
        """Decompose call option price.

        (S - K)^+ = S * I_{S > K} - K * I_{S > K}.
        """
        c = call.Call(self.rate, self.vol, self.strike, self.expiry_idx,
                      self.event_grid)
        b1 = binary.BinaryAssetCall(self.rate, self.vol, self.event_grid, self.strike, self.expiry_idx)
        b2 = binary.BinaryCashCall(self.rate, self.vol, self.event_grid, self.strike, self.expiry_idx)
        call_decomposed = b1.price(self.spot, self.time_idx) \
            - self.strike * b2.price(self.spot, self.time_idx)
        diff = np.abs(c.price(self.spot, self.time_idx) - call_decomposed)
        self.assertTrue(np.max(diff) < 1.0e-12)


if __name__ == '__main__':

    # Plot call option price and decompose into binary asset/cash call prices
    rate = 0.05
    vol = 0.2

    time = 0
    time_idx = 0
    expiry = 2
    expiry_idx = 1
    event_grid = np.array([time, expiry])

    strike = 50

    spot = np.arange(2, 100, 2) * 1.0

    c = call.Call(rate, vol, strike, expiry_idx, event_grid)
    b1 = binary.BinaryAssetCall(rate, vol, event_grid, strike, expiry_idx)
    b2 = binary.BinaryCashCall(rate, vol, event_grid, strike, expiry_idx)

    plt.plot(spot, c.payoff(spot), "-k", label="Option payoff")
    plt.plot(spot, c.price(spot, time_idx), "-ob", label="Call")
    plt.plot(spot, b1.price(spot, time_idx), "-r", label="Binary asset call")
    plt.plot(spot, strike * b2.price(spot, time_idx), "-g", label="Binary cash call")
    call_price_decomposed = \
        b1.price(spot, time_idx) - strike * b2.price(spot, time_idx)
    plt.plot(spot, call_price_decomposed, "-y", label="Combine binaries")
    plt.title("Call option, Black-Scholes model")
    plt.xlabel("Stock price")
    plt.ylabel("Call price")
    plt.legend()
    plt.show()

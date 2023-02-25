import math
import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import call
from models.black_scholes import put


class Parity(unittest.TestCase):
    """Test call-put parity in Black-Scholes model."""

    def test_1(self):
        rate = 0.05
        vol = 0.2
        time = 0
        time_idx = 0
        expiry = 2
        expiry_idx = 1
        event_grid = np.array([time, expiry])
        strike = 50
        spot = np.arange(2, 100, 2) * 1.0
        c = call.Call(rate, vol, event_grid, strike, expiry_idx)
        p = put.Put(rate, vol, event_grid, strike, expiry_idx)
        discount = math.exp(-rate * (expiry - time))
        lhs = c.price(spot, time_idx) - p.price(spot, time_idx)
        rhs = spot - strike * discount
        self.assertTrue(np.max(np.abs(lhs - rhs)) < 1.0e-12)


if __name__ == '__main__':

    unittest.main()

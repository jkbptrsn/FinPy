import math
import unittest

from matplotlib import pyplot as plt
import numpy as np

from models.black_scholes import call
from models.black_scholes import put


class Parity(unittest.TestCase):
    """Test call-put parity in Black-Scholes model."""

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

    def test_1(self):
        c = call.Call(self.rate, self.vol, self.strike, self.expiry_idx,
                      self.event_grid)
        p = put.Put(self.rate, self.vol, self.strike, self.expiry_idx,
                    self.event_grid)
        lhs = c.price(self.spot, self.time_idx) \
            - p.price(self.spot, self.time_idx)
        discount = math.exp(-self.rate * (self.expiry - self.time))
        rhs = self.spot - self.strike * discount
        self.assertTrue(np.max(np.abs(lhs - rhs)) < 1.0e-12)


if __name__ == '__main__':
    unittest.main()

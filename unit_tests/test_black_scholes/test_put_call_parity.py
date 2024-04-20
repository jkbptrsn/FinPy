import math
import unittest

import numpy as np

from models.black_scholes import european_option as option


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

    def test_call_put_parity(self):
        c = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid,
            type_="Call")
        p = option.EuropeanOption(
            self.rate, self.vol, self.strike, self.expiry_idx, self.event_grid,
            type_="Put")
        lhs = c.price(self.spot, self.time_idx) \
            - p.price(self.spot, self.time_idx)
        discount = math.exp(-self.rate * (self.expiry - self.time))
        rhs = self.spot - self.strike * discount
        self.assertTrue(np.max(np.abs(lhs - rhs)) < 2.0e-14)

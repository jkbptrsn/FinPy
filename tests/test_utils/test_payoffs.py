import unittest

import numpy as np

from utils import payoffs


class CallOptions(unittest.TestCase):
    """Call option payoffs."""

    def setUp(self) -> None:
        self.spot = np.arange(0, 100)
        self.strike = 50

    def test_decompose(self):
        """Decompose call option payoff.

        (S - K)^+ = S * I_{S > K} - K * I_{S > K}
        """
        call = payoffs.call(self.spot, self.strike)
        binary_asset = payoffs.binary_asset_call(self.spot, self.strike)
        binary_cash = payoffs.binary_cash_call(self.spot, self.strike)
        assertion = np.all(call == binary_asset - self.strike * binary_cash)
        self.assertTrue(assertion)


class PutOptions(unittest.TestCase):
    """Put option payoffs."""

    def setUp(self) -> None:
        self.spot = np.arange(0, 100)
        self.strike = 50

    def test_decompose(self):
        """Decompose put option payoff.

        (K - S)^+ = K * I_{S < K} - S * I_{S < K}
        """
        put = payoffs.put(self.spot, self.strike)
        binary_asset = payoffs.binary_asset_put(self.spot, self.strike)
        binary_cash = payoffs.binary_cash_put(self.spot, self.strike)
        assertion = np.all(put == self.strike * binary_cash - binary_asset)
        self.assertTrue(assertion)

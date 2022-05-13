import models.black_scholes.call as call

import unittest


class CallOption(unittest.TestCase):

    def test_call(self):
        c = call.Call(0.05, 0.2, 50, 2)
        self.assertTrue(c.price(50, 0) > 0)


if __name__ == '__main__':
    unittest.main()

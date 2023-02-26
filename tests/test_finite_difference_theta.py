import unittest

import numerical_methods.finite_difference.theta as theta


class Theta(unittest.TestCase):

    def test_misc(self):
        solver = theta.Theta1D(0, 100, 11)
        self.assertTrue(solver.xmin == 0)
        self.assertTrue(solver.xmax == 100)
        self.assertTrue(solver.nstates == 11)
        self.assertAlmostEqual(solver.dx, 10, 10)


if __name__ == '__main__':
    unittest.main()

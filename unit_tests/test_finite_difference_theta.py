import unittest

import numerical_methods.finite_difference.theta as theta


class Theta(unittest.TestCase):

    def test_misc(self):
        solver = theta.Theta(0, 100, 11, 1.3, theta=0.8)
        self.assertTrue(solver.xmin == 0)
        self.assertTrue(solver.xmax == 100)
        self.assertTrue(solver.nstates == 11)
        self.assertTrue(solver.dt == 1.3)
        self.assertTrue(solver.theta == 0.8)
        self.assertAlmostEqual(solver.dx, 10, 10)


if __name__ == '__main__':
    unittest.main()

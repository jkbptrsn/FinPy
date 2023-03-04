import unittest

import numpy as np

import numerical_methods.finite_difference.theta as theta


class Theta(unittest.TestCase):

    def test_misc(self):
        grid = 10 * np.arange(11)
        solver = theta.ThetaBase(grid)
        self.assertTrue(solver.xmin == 0)
        self.assertTrue(solver.xmax == 100)
        self.assertTrue(solver.nstates == 11)


if __name__ == '__main__':
    unittest.main()

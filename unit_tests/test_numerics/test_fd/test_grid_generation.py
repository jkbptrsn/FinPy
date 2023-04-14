import unittest

from matplotlib import pyplot as plt
import numpy as np

from numerics.fd import grid_generation as grid


class TestSomething(unittest.TestCase):

    def test_this(self):
        print("test_template.py")
        self.assertEqual(2, 2)


if __name__ == '__main__':

    # K.J. In 'T Hout and S. Foulon, 2010.
    # Grid in s-direction.
    s_min = 0
    s_max = 800
    s_states = 30
    s_center = 100
    shape = 5
    uniform, s_grid = grid.hyperbolic(s_min, s_max, s_states, s_center, shape)
    print(s_grid)
    plt.plot(uniform, s_grid, "o")
    plt.ylabel("S variable")
    plt.show()

    # K.J. In 'T Hout and S. Foulon, 2010.
    # Grid in v-direction.
    v_min = 0
    v_max = 1
    v_states = 15
    v_center = 0
    shape = 1 / 500
    uniform, v_grid = grid.hyperbolic(v_min, v_max, v_states, v_center, shape)
    print(v_grid)
    plt.plot(uniform, v_grid, "o")
    plt.ylabel("v variable")
    plt.show()

    s_scatter = np.zeros(s_grid.size * v_grid.size)
    v_scatter = np.zeros(s_grid.size * v_grid.size)
    for idx, s in enumerate(s_grid):
        idx_1 = v_grid.size * idx
        idx_2 = v_grid.size * (idx + 1)
        s_scatter[idx_1:idx_2] = s
        v_scatter[idx_1:idx_2] = v_grid
    plt.scatter(s_scatter, v_scatter, marker=".")
    plt.xlabel("S variable")
    plt.ylabel("v variable")
    plt.show()

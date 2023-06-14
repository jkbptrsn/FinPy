import math

from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize


def power_functions_1st(x, a, b):
    return a + b * x


def power_functions_2nd(x, a, b, c):
    return a + b * x + c * x ** 2


def power_functions_3rd(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


def power_functions_4th(x, a, b, c, d, e):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


def power_functions_5th(x, a, b, c, d, e, f):
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5


def regression(fn, x, y):
    return optimize.curve_fit(fn, xdata=x, ydata=y)[0]


def price_american_put(paths):
    """

    Args:
        paths: Monte-Carlo paths.

    Returns:
        American option price.
    """

    fn = power_functions_3rd

    exercise_index = -np.ones(paths.shape[1], dtype=int)
    con_value = np.zeros(paths.shape[1])

    x_grid = np.linspace(10, 80)

    for n in range(paths.shape[0] - 1, -1, -1):

        if n % 10 == 0:
            # Immediate exercise value.
            exercise_value = np.maximum(40 - paths[n, :], 0)

            # Expected continuation value.
            parms = regression(fn, paths[n, :], con_value)
#            plt.plot(paths[n, :], con_value, "ob")
#            plt.plot(x_grid, fn(x_grid, *parms), "-r")
#            plt.pause(0.5)
#            plt.show()
#            plt.cla()

            exp_con_value = fn(paths[n, :], *parms)
            exp_con_value = np.maximum(exp_con_value, 0)

#            tmp = np.maximum(exercise_value - con_value, 0)
            tmp = np.maximum(exercise_value - exp_con_value, 0)

            for idx in np.nonzero(tmp):
                exercise_index[idx] = n
                con_value[idx] = exercise_value[idx]

            # Discounting.
            discount_factor = math.exp(-0.06 * 0.02)
            con_value *= discount_factor

    mc_average = 0
    for m in range(paths.shape[1]):
        idx = exercise_index[m]
        if idx != -1:
            exercise_value = np.max(40 - paths[exercise_index[m], m], 0)
            # Time to exercise
            time_exercise = exercise_index[m] / 500

            mc_average += exercise_value * math.exp(-0.06 * time_exercise)

    # print(exercise_index)

    mc_average /= paths.shape[1]

    return mc_average

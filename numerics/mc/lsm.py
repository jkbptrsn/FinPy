import math

from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize


def polynomial_1st(x, a, b):
    """First order polynomial."""
    return a + b * x


def polynomial_2nd(x, a, b, c):
    """Second order polynomial."""
    return a + b * x + c * x ** 2


def polynomial_3rd(x, a, b, c, d):
    """Third order polynomial."""
    return a + b * x + c * x ** 2 + d * x ** 3


def polynomial_4th(x, a, b, c, d, e):
    """Fourth order polynomial."""
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


def polynomial_5th(x, a, b, c, d, e, f):
    """Fifth order polynomial."""
    return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5


def regression(fn, x, y):
    """Fit function fn to data set (x,y)."""
    return optimize.curve_fit(fn, xdata=x, ydata=y)[0]


def price_american_put(paths,
                       event_grid,
                       exercise_grid):
    """Pricing American put option using Longstaff-Schwartz method.

    Args:
        paths: Monte-Carlo paths.
        event_grid: ...
        exercise_grid: Exercise event indices on event grid.

    Returns:
        American option price.
    """

    plot_regression = False

    # Regression function object.
    fn = polynomial_3rd

    # Index of exercise (on event grid) for each path.
    exercise_index = -np.ones(paths.shape[1], dtype=int)
    # Continuation value for each path.
    con_value = np.zeros(paths.shape[1])

    n_old = None
    for n in exercise_grid:

            # Discounting.
            if n_old is None:
                discount_factor = 1
                n_old = n
            else:
                dt = event_grid[n_old] - event_grid[n]
                discount_factor = math.exp(-0.06 * dt)
                n_old = n
            con_value *= discount_factor

            # Immediate exercise value.
            exercise_value = np.maximum(40 - paths[n, :], 0)

            # Expected continuation value.
            parms = regression(fn, paths[n, :], con_value)

            if plot_regression:
                x_grid = np.linspace(10, 80)
                plt.plot(paths[n, :], con_value, "ob")
                plt.plot(x_grid, fn(x_grid, *parms), "-r")
                plt.pause(0.5)
                plt.cla()

            exp_con_value = fn(paths[n, :], *parms)
            exp_con_value = np.maximum(exp_con_value, 0)

#            tmp = np.maximum(exercise_value - con_value, 0)
            tmp = np.maximum(exercise_value - exp_con_value, 0)

            for idx in np.nonzero(tmp):
                exercise_index[idx] = n
                con_value[idx] = exercise_value[idx]

    mc_average = 0
    for m in range(paths.shape[1]):
        idx = exercise_index[m]
        if idx != -1:
            exercise_value = np.max(40 - paths[idx, m], 0)
            # Time to exercise
            time_exercise = event_grid[idx]

            mc_average += exercise_value * math.exp(-0.06 * time_exercise)

    mc_average /= paths.shape[1]

    return mc_average

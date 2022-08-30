import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import unittest

from tests.test_hull_white import input
from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from utils import misc


if __name__ == "__main__":

    # Event dates in year fractions.
    event_grid = np.arange(0, 11, 10)

    # Speed of mean reversion.
    kappa_const = 0.015

    # Volatility.
    vol_const = 0.005

    # Speed of mean reversion strip.
    kappa = \
        np.array([np.array([2, 3, 7]), kappa_const * np.array([2, 1, 2])])
    kappa = misc.DiscreteFunc("kappa", kappa[0], kappa[1])

    # Volatility strip.
    vol = np.array([np.arange(10),
                    vol_const * np.array([1, 2, 3, 1, 1, 5, 6, 6, 3, 3])])
    vol = misc.DiscreteFunc("vol", vol[0], vol[1])

    # Discount curve.
    forward_rate = np.array([-0.005, 0.02])
    discount_curve = np.exp(-forward_rate * event_grid)
    discount_curve = \
        misc.DiscreteFunc("discount curve", event_grid, discount_curve)

    # Analytical results.
    price_a = discount_curve.values

    # SDE object.
    hw = sde.SDE(kappa, vol, event_grid, int_step_size=1 / 52)

    # Number of repetitions
    n_reps = 50

    # Number of Monte-Carlo paths.
    n_paths = 2 ** 16
    print(f"Number of repetitions: {n_reps}")
    print(f"Number of paths: {n_paths}")

    errors = np.zeros((3, event_grid.size))

    # Random number generator.
    rng = np.random.default_rng(0)

    # Ordinary Monte-Carlo
    for n in range(n_reps):
        rate, discount = hw.paths(0, n_paths, rng=rng, antithetic=False)
        discount = sde.discount_adjustment(discount, discount_curve)
        price_n = discount.sum(axis=1) / n_paths
        diff = np.abs((price_n - price_a) / price_a)
        errors[0, :] += diff

    # Antithetic sampling
    for n in range(n_reps):
        rate, discount = hw.paths(0, n_paths, rng=rng, antithetic=True)
        discount = sde.discount_adjustment(discount, discount_curve)
        price_n = discount.sum(axis=1) / n_paths
        diff = np.abs((price_n - price_a) / price_a)
        errors[1, :] += diff

    # Antithetic sampling
    for n in range(n_reps):
        rate, discount = hw.paths_sobol_test(0, n_paths, rng=rng, antithetic=True, n_rep=n)
        discount = sde.discount_adjustment(discount, discount_curve)
        price_n = discount.sum(axis=1) / n_paths
        diff = np.abs((price_n - price_a) / price_a)
        errors[2, :] += diff

    errors /= n_reps
    print("\nAverage relative error:\n")
    print(errors)

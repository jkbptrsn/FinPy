import datetime
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy

from unit_tests.test_hull_white import input
from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from utils import mbb
from utils import misc


if __name__ == "__main__":

    start_init = time.perf_counter()

    # Speed of mean reversion strip.
    kappa = input.kappa_strip

    # Volatility strip.
    vol = input.vol_strip

    # Event dates in year fractions.
    event_grid = input.time_grid_ext

    # Discount curve.
    discount_curve = input.disc_curve_ext

    # time_grid_plot = 0.01 * np.arange(3001)
    # plt.plot(time_grid_plot, kappa.interpolation(time_grid_plot))
    # plt.show()
    #
    # time_grid_plot = 0.01 * np.arange(3001)
    # plt.plot(time_grid_plot, vol.interpolation(time_grid_plot))
    # plt.show()
    #
    # time_grid_plot = 0.01 * np.arange(3001)
    # plt.plot(time_grid_plot, discount_curve.interpolation(time_grid_plot))
    # plt.plot(discount_curve.time_grid, discount_curve.values)
    # plt.show()

    # Array with number of MC paths for each convergence test.
    n_paths_array = np.array([2 ** e for e in range(10, 13, 2)])
    # Number of repetitions per test.
    n_reps = 100

    # Integration step size
    int_step_size = 1 / 365

    # SDE object.
    hw = sde.SDE(kappa, vol, event_grid, int_step_size=int_step_size)

    # Random number generator.
    rng = np.random.default_rng(0)

    # Coupon
    coupon = 0.03
    coupon_tau = coupon * np.diff(event_grid)

    # Analytical results.
    price_a = np.sum(coupon_tau * discount_curve.values[1:]) \
        + discount_curve.values[-1]
    price_a *= 100

    average_price = np.zeros((2, n_paths_array.size))
    rms_errors = np.zeros((2, n_paths_array.size))

    end_init = time.perf_counter()
    print(f"Timing initialization: "
          f"{datetime.timedelta(seconds=end_init - start_init)}")

    for idx, n_paths in enumerate(n_paths_array):

        start_rep = time.perf_counter()

        for _ in range(n_reps):

            # Ordinary Monte-Carlo.
            rate, discount = hw.paths(0, n_paths, rng=rng, antithetic=False)
            discount = sde.discount_adjustment(discount, discount_curve)
            price_n = discount.sum(axis=1) / n_paths
            price_n = np.sum(coupon_tau * price_n[1:]) + price_n[-1]
            price_n *= 100
            average_price[0, idx] += price_n
            rms_errors[0, idx] += ((price_n - price_a) / price_a) ** 2

            # Antithetic sampling.
            rate, discount = hw.paths(0, n_paths, rng=rng, antithetic=True)
            discount = sde.discount_adjustment(discount, discount_curve)
            price_n = discount.sum(axis=1) / n_paths
            price_n = np.sum(coupon_tau * price_n[1:]) + price_n[-1]
            price_n *= 100
            average_price[1, idx] += price_n
            rms_errors[1, idx] += ((price_n - price_a) / price_a) ** 2

        end_rep = time.perf_counter()
        # Average execution time
        average_time = (end_rep - start_rep) / n_reps

        # Average price.
        average_price[:, idx] /= n_reps
        # Root mean square error.
        rms_errors[:, idx] = np.sqrt(rms_errors[:, idx] / n_reps)

        print(f"\nTest {idx + 1} with number of paths = {n_paths}:")
        print(f"* 'Analytical' result: {price_a}")
        print(f"* Ordinary Monte-Carlo: {average_price[0, idx]}. "
              f"Abs diff = {abs(average_price[0, idx] - price_a)}")
        print(f"* Antithetic sampling: {average_price[1, idx]}. "
              f"Abs diff = {abs(average_price[1, idx] - price_a)}")
        print(f"* Average execution time: "
              f"{datetime.timedelta(seconds=average_time)}")

    reg = \
        scipy.stats.linregress(np.log(n_paths_array), np.log(rms_errors[0, :]))
    reg_line = n_paths_array ** reg.slope * np.exp(reg.intercept)
    plt.plot(n_paths_array, rms_errors[0, :], "ob")
    plt.plot(n_paths_array, reg_line, "-b", linewidth=0.6)

    reg = \
        scipy.stats.linregress(np.log(n_paths_array), np.log(rms_errors[1, :]))
    reg_line = n_paths_array ** reg.slope * np.exp(reg.intercept)
    plt.plot(n_paths_array, rms_errors[1, :], "or")
    plt.plot(n_paths_array, reg_line, "-r", linewidth=0.6)

    plt.xlabel("Number of paths")
    plt.ylabel("Root mean square error")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    end_time = time.perf_counter()
    print(f"\nTotal execution time: "
          f"{datetime.timedelta(seconds=end_time-start_init)}")

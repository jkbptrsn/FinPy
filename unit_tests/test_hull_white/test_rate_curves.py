import datetime
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np

from unit_tests.test_hull_white import input
from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from utils import misc


if __name__ == "__main__":

    # Yield curve on "bond grid".
    yield_curve = input.yield_curve_ext

    # Discount curve on "bond grid".
    discount_curve = input.disc_curve_ext
    event_grid = discount_curve.time_grid

    # Instantaneous forward rate curve on "bond grid".
    forward_rate = input.forward_rate_ext

#    plt.plot(yield_curve.time_grid, 100 * yield_curve.values)
#    plt.plot(discount_curve.time_grid, discount_curve.values)
#    plt.plot(forward_rate.time_grid, 100 * forward_rate.values)
#    plt.show()

    # Speed of mean reversion strip.
    kappa = input.kappa_strip
#    kappa_int = kappa.interpolation(event_grid)
#    plt.plot(event_grid, kappa_int)
#    plt.show()

    # Volatility strip.
    vol = input.vol_strip
#    vol_int = vol.interpolation(event_grid)
#    plt.plot(event_grid, vol_int)
#    plt.show()

    # Zero-coupon bond object
    int_step_size = 1 / 52
    maturity_idx = event_grid.size - 1
    bond = zero_coupon_bond.ZCBond(kappa, vol,
                                   discount_curve, discount_curve.time_grid,
                                   maturity_idx, int_step_size)

    # Construct yield curve at event_idx with x_state = spot
    event_idx = 3
    print(f"Term {event_idx} at time {event_grid[event_idx]}")
    spot = np.array([-0.005, 0, 0.005])
    maturity_indices = np.arange(event_idx + 1, event_grid.size)
    time_vector = event_grid[event_idx + 1:]
    event_slice = event_grid[event_idx:]
    tau_cumulative = np.cumsum(np.diff(event_slice))
    price_vector = bond.price_vector(spot, event_idx, maturity_indices)

    plots = plt.plot(forward_rate.time_grid, 100 * forward_rate.values,
                     "-.k", label="f(0,t)")
    plot_style = ["-b", "-r", "-k"]
    dot_style = ["ob", "or", "ok"]
    labels = ["x = -50 bps", "x = 0", "x = 50 bps"]
    for n in range(spot.size):
        rate_vector = -np.log(price_vector[:, n]) / tau_cumulative
        if n == 0:
            plots += plt.plot(time_vector, 100 * rate_vector,
                             plot_style[n], label=labels[n])
            plt.plot(time_vector[0], 100 * rate_vector[0], dot_style[n])
        elif n == 1:
            time_center = np.append(event_grid[event_idx], time_vector)
            rate_center = np.append(forward_rate.values[event_idx - 1], rate_vector)
            plots += plt.plot(time_center, 100 * rate_center,
                              plot_style[n], label=labels[n])
            plt.plot(time_center[1], 100 * rate_center[1], dot_style[n])
        else:
            plots += plt.plot(time_vector, 100 * rate_vector,
                              plot_style[n], label=labels[n])
            plt.plot(time_vector[0], 100 * rate_vector[0], dot_style[n])
    plt.xlabel(f"t [years]")
    plt.ylabel(f"Yield [%]")
    plt.legend(plots, [plot.get_label() for plot in plots], loc=4)
    plt.show()

import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
import unittest

from unit_tests.test_hull_white import input
from models.hull_white import call_option
from models.hull_white import sde
from models.hull_white import zero_coupon_bond
from utils import misc


if __name__ == "__main__":

    # Event dates in year fractions.
    event_grid = np.arange(0, 11, 5)

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
    forward_rate = np.array([-0.005, 0.023, 0.02])
    discount_curve = np.exp(-forward_rate * event_grid)
    discount_curve = \
        misc.DiscreteFunc("discount curve", event_grid, discount_curve)


    # 30Y mortgage bond
    kappa = input.kappa_strip
    vol = input.vol_strip
    event_grid = input.time_grid_ext
    discount_curve = input.disc_curve_ext


    # Analytical results.
    price_a = discount_curve.values

    # Integration step size
    int_step_size = 1 / 365

    # SDE object.
    hw = sde.SDE(kappa, vol, event_grid, int_step_size=int_step_size)

    # Coupon.
    coupon = 0.03

    # Zero-coupon bond object.
    bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve,
                                   event_grid, event_grid.size - 1,
                                   int_step_size=int_step_size)

    # Call option object.
    strike = 0.5
    expiry_idx = 60
    maturity_idx = event_grid.size - 1
    call = call_option.Call(kappa, vol, discount_curve, event_grid,
                            strike, expiry_idx, maturity_idx,
                            int_step_size=int_step_size)

    # Instrument type.
#    instrument = "ZCBond"
#    instrument = "Call"
    instrument = "Coupon bond"

    # Analytical results.
    price_a = None
    if instrument == "ZCBond":
        price_a = discount_curve.values[-1]
    elif instrument == "Call":
        price_a = call.price(0, 0)
    elif instrument == "Coupon bond":
        price_a = np.sum(coupon * discount_curve.values[1:]) \
                  + discount_curve.values[-1]

    print("Instrument type " + instrument)
    print(f"Analytical price: {price_a}")

    n_expos = 5
    grid_plot = np.zeros((n_expos + 1) // 2)
    errors_plot = np.zeros((3, (n_expos + 1) // 2))

    # Number of repetitions
    n_reps = 50
    print(f"Number of repetitions: {n_reps}")

    # Random number generator.
    rng = np.random.default_rng(0)

    for idx, expo in enumerate(range(10, 10 + n_expos, 2)):

        # Number of MC paths.
        n_paths = 2 ** expo

        print(f"Number of paths: {n_paths}")

        errors = np.zeros(3)

        price_vector = np.zeros(3)

        for n in range(n_reps):

            # Ordinary Monte-Carlo.
            rate, discount = hw.paths(0, n_paths, rng=rng, antithetic=False)
            discount = sde.discount_adjustment(discount, discount_curve)
            if instrument == "ZCBond":
                price_vector[0] = np.sum(discount[-1]) / n_paths
            elif instrument == "Call":
                bond_price = bond.price(rate[expiry_idx], expiry_idx)
                pay_off = np.maximum(bond_price - strike, 0)
                price_vector[0] = \
                    np.sum(discount[expiry_idx] * pay_off) / n_paths
            elif instrument == "Coupon bond":
                discount_n = discount.sum(axis=1) / n_paths
                price_vector[0] = \
                    np.sum(coupon * discount_n[1:]) + discount_n[-1]

            # Antithetic sampling.
            rate, discount = hw.paths(0, n_paths, rng=rng, antithetic=True)
            discount = sde.discount_adjustment(discount, discount_curve)
            if instrument == "ZCBond":
                price_vector[1] = np.sum(discount[-1]) / n_paths
            elif instrument == "Call":
                bond_price = bond.price(rate[expiry_idx], expiry_idx)
                pay_off = np.maximum(bond_price - strike, 0)
                price_vector[1] = \
                    np.sum(discount[expiry_idx] * pay_off) / n_paths
            elif instrument == "Coupon bond":
                discount_n = discount.sum(axis=1) / n_paths
                price_vector[1] = \
                    np.sum(coupon * discount_n[1:]) + discount_n[-1]

            # Sobol sequence.
            sobol_gen = misc.sobol_init(event_grid.size)
            sobol_uni = sobol_gen.random(n_paths)
            sobol_norm = scipy.stats.norm.ppf(sobol_uni)
            rate, discount = hw.paths_sobol_test(0, n_paths, sobol_norm)
            discount = sde.discount_adjustment(discount, discount_curve)
            if instrument == "ZCBond":
                price_vector[2] = np.sum(discount[-1]) / n_paths
            elif instrument == "Call":
                bond_price = bond.price(rate[expiry_idx], expiry_idx)
                pay_off = np.maximum(bond_price - strike, 0)
                price_vector[2] = \
                    np.sum(discount[expiry_idx] * pay_off) / n_paths
            elif instrument == "Coupon bond":
                discount_n = discount.sum(axis=1) / n_paths
                price_vector[2] = \
                    np.sum(coupon * discount_n[1:]) + discount_n[-1]

            errors += np.abs((price_vector - price_a) / price_a)
        errors /= n_reps
        print("\nAverage relative error:\n")
        print(errors)

        errors_plot[:, idx] = errors
        grid_plot[idx] = n_paths

    reg = scipy.stats.linregress(np.log(grid_plot), np.log(errors_plot[0, :]))
    reg_line = grid_plot ** reg.slope * np.exp(reg.intercept)
    p1 = plt.plot(grid_plot, errors_plot[0, :],
                  "ob", label=f"Ordinary MC: {reg.slope:0.2f}")
    plt.plot(grid_plot, reg_line, "-b", linewidth=0.6)
    reg = scipy.stats.linregress(np.log(grid_plot), np.log(errors_plot[1, :]))
    reg_line = grid_plot ** reg.slope * np.exp(reg.intercept)
    p2 = plt.plot(grid_plot, errors_plot[1, :],
                  "or", label=f"Antithetic sampling: {reg.slope:0.2f}")
    plt.plot(grid_plot, reg_line, "-r", linewidth=0.6)
    reg = scipy.stats.linregress(np.log(grid_plot), np.log(errors_plot[2, :]))
    reg_line = grid_plot ** reg.slope * np.exp(reg.intercept)
    p3 = plt.plot(grid_plot, errors_plot[2, :],
                  "ok", label=f"Sobol sequence: {reg.slope:0.2f}")
    plt.plot(grid_plot, reg_line, "-k", linewidth=0.6)
    plt.xlabel("Number of paths")
    plt.ylabel("Absolute relative error")
    plt.xscale("log")
    plt.yscale("log")
    plots = p1 + p2 + p3
    plt.legend(plots, [plot.get_label() for plot in plots])
    plt.show()

    grid_log = np.log(grid_plot)
    error_log = np.log(errors_plot[0, :])

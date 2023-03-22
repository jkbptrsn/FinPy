import datetime
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np

from unit_tests.test_hull_white import input
from models.hull_white import call_option
from models.hull_white import caplet
from models.hull_white import put_option
from models.hull_white import sde
from models.hull_white import swap
from models.hull_white import swaption
from models.hull_white import zero_coupon_bond
from utils import misc


def execution_time(func):
    """Decorator for timing function execution."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        delta = datetime.timedelta(seconds=end - start)
        print(f"Execution time for {func.__name__}: {delta}")
        return result
    return wrapper


class Convergence(unittest.TestCase):

    def test_convergence(self):
        """..."""
        pass


def mc_simulation(instrument: str,
                  price_a: float,
                  hw: sde.SDE,
                  bond: zero_coupon_bond.ZCBond,
                  instru_obj,
                  discount_curve: misc.DiscreteFunc,
                  rng: np.random.Generator,
                  n_bins: int,
                  price_min: float,
                  price_max: float,
                  n_reps1: int,
                  n_reps2: int,
                  n_paths_array: np.ndarray,
                  antithetic: bool = False) -> tuple:
    """
    Collect data from Monte-Carlo simulations...

    Args:
        instrument: Name of instrument.
        price_a: Analytical price.
        hw: Hull-White SDE object.
        bond: Zero-coupon bond object.
        instru_obj: Instrument object.
        discount_curve: Discount curve represented on event-grid.
        rng: Random number generator.
        n_bins: Number of bins on price grid.
        price_min: Minimum value on price grid.
        price_max: Maximum value on price grid.
        n_reps1: Number of repetitions in outer for-loop.
        n_reps2: Number of repetitions in inner for-loop.
        n_paths_array: Array of number of MC paths per simulation.
        antithetic: Antithetic sampling.

    Returns:
        ...
    """
    # Price grid (each grid point is centered in the corresponding bin).
    price_delta = (price_max - price_min) / n_bins
    grid = price_delta * (np.arange(n_bins) + 0.5) + price_min
    # Histogram of instrument price.
    histogram = np.zeros((n_bins, n_paths_array.size))
    # Mean of all scenarios used in construction of each histogram.
    mean = np.zeros(n_paths_array.size)
    # Mean based on integration of histogram.
    h_mean = np.zeros(n_paths_array.size)
    # Standard deviation based on integration of histogram.
    h_std = np.zeros(n_paths_array.size)
    # Relative error, i.e. <| (price_n - price_a) / price_a |>.
    r_error = np.zeros(n_paths_array.size)
    # Relative error for one simulation.
    r_error_path = np.zeros(n_paths_array.size)
    # Loop over number-of-paths-per-simulation.
    for idx, n_paths in enumerate(n_paths_array):
        # Outer repetition loop
        for rep1_idx in range(n_reps1):
            price_array = np.zeros(n_reps2)
            # Inner repetition loop
            for rep2_idx in range(n_reps2):
                # Pseudo short rate and discount factor.
                rate, discount = \
                    hw.paths(0, n_paths, rng=rng, antithetic=antithetic)
                # Adjust pseudo discount factor.
                discount = sde.discount_adjustment(discount, discount_curve)
                # Monte-Carlo estimate of instrument price.
                if instrument == "ZCBond":
                    # Zero-coupon bond price at time zero.
                    # VanillaBond matures at last event.
                    index = bond.maturity_idx
                    price_tmp = discount[index]
                elif instrument == "Call" or instrument == "Put":
                    # Index corresponding to expiry.
                    index = instru_obj.expiry_idx
                    # Zero-coupon bond price at expiry.
                    # VanillaBond matures at last event.
                    bond_price = bond.price(rate[index], index)
                    # Call/put option price.
                    if instrument == "Call":
                        payoff = np.maximum(bond_price - instru_obj.strike, 0)
                    else:
                        payoff = np.maximum(instru_obj.strike - bond_price, 0)
                    price_tmp = discount[index] * payoff
                elif instrument == "Swap":
                    # Simple forward rate at each event.
                    simple_forward_rate = np.zeros(discount.shape)
                    for event_idx, tau in enumerate(np.diff(hw.event_grid)):
                        discount_forward = swap_obj.zcbond._calc_price(
                            rate[event_idx], event_idx, event_idx + 1)
                        simple_forward_rate[event_idx] = \
                            (1 / discount_forward - 1) / tau
                    # Swap price.
                    swap_price_n = discount[1:] \
                        * (simple_forward_rate[:-1] - swap_obj.fixed_rate)
                    swap_price_n = \
                        swap_price_n.transpose() * np.diff(hw.event_grid)
                    price_tmp = np.sum(swap_price_n.transpose(), axis=0)
                else:
                    raise TypeError(f"Instrument unknown: {instrument}")
                # Monte-Carlo estimate for repetition rep2_idx.
                price_array[rep2_idx] = np.sum(price_tmp) / n_paths
            # Relative error.
            r_error[idx] += np.sum(np.abs((price_array - price_a) / price_a))
            if rep1_idx == 0:
                r_error_path[idx] = \
                    np.abs((price_array[0] - price_a) / price_a)
            # Normalized histogram of instrument price.
            hist_tmp = np.histogram(price_array, bins=n_bins,
                                    range=(price_min, price_max), density=True)
            histogram[:, idx] += hist_tmp[0]
            # Mean of all scenarios.
            mean[idx] += price_array.sum()
        # Average histogram for n_reps1 * n_reps2 repetitions of MC
        # simulation.
        histogram[:, idx] /= n_reps1
        # RMean of all scenarios.
        mean[idx] /= (n_reps1 * n_reps2)
        # Mean based on integration of histogram.
        integrand = histogram[:, idx] * grid
        h_mean[idx] = np.sum(integrand) * price_delta
        # Standard deviation based on integration of histogram.
        integrand = histogram[:, idx] * (grid - h_mean[idx]) ** 2
        h_std[idx] = np.sqrt(np.sum(integrand) * price_delta)
        # Relative error.
        r_error[idx] /= (n_reps1 * n_reps2)
    return grid, histogram, mean, h_mean, h_std, r_error, r_error_path


@execution_time
def plot_type_1(instrument: str,
                price_a: float,
                hw: sde.SDE,
                bond: zero_coupon_bond.ZCBond,
                instru_obj,
                discount_curve: misc.DiscreteFunc,
                rng: np.random.Generator,
                n_bins: int,
                price_min: float,
                price_max: float,
                n_reps1: int,
                n_reps2: int):
    """
    Plot...

    Args:
        instrument: Name of instrument.
        price_a: Analytical price.
        hw: Hull-White SDE object.
        bond: Zero-coupon bond object.
        instru_obj: Instrument object.
        discount_curve: Discount curve represented on event-grid.
        rng: Random number generator.
        n_bins: Number of bins on price grid.
        price_min: Minimum value on price grid.
        price_max: Maximum value on price grid.
        n_reps1: Number of repetitions in outer for-loop.
        n_reps2: Number of repetitions in inner for-loop.
    """
    # Number of MC paths per simulation.
    n_paths = np.array([100, 400, 1600])
    do_anti = [False, True]
    for anti in do_anti:
        grid, h_matrix, mean, h_mean, h_std, r_error, r_error_path = \
            mc_simulation(instrument, price_a, hw, bond, instru_obj,
                          discount_curve, rng,
                          n_bins, price_min, price_max,
                          n_reps1, n_reps2, n_paths,
                          antithetic=anti)
        p1 = plt.plot(grid, h_matrix[:, 0], "-b", label=f"{n_paths[0]} paths")
        p2 = plt.plot(grid, h_matrix[:, 1], "-r", label=f"{n_paths[1]} paths")
        p3 = plt.plot(grid, h_matrix[:, 2], "-k", label=f"{n_paths[2]} paths")
        plots = p1 + p2 + p3
        plt.axvline(price_a, c="grey", ls=":")
        plt.xlabel("Price")
        plt.ylabel("Probability density")
        plt.legend(plots, [plot.get_label() for plot in plots])
        plt.savefig(f"density_2d_{instrument}_{anti}.png")
        plt.clf()


@execution_time
def plot_type_2(instrument: str,
                price_a: float,
                hw: sde.SDE,
                bond: zero_coupon_bond.ZCBond,
                instru_obj,
                discount_curve: misc.DiscreteFunc,
                rng: np.random.Generator,
                n_bins: int,
                price_min: float,
                price_max: float,
                n_reps1: int,
                n_reps2: int):
    """
    Plot...

    Args:
        instrument: Name of instrument.
        price_a: Analytical price.
        hw: Hull-White SDE object.
        bond: Zero-coupon bond object.
        instru_obj: Instrument object.
        discount_curve: Discount curve represented on event-grid.
        rng: Random number generator.
        n_bins: Number of bins on price grid.
        price_min: Minimum value on price grid.
        price_max: Maximum value on price grid.
        n_reps1: Number of repetitions in outer for-loop.
        n_reps2: Number of repetitions in inner for-loop.
    """
    # Number of MC paths per simulation.
    n_paths = np.array([100, 400, 1600])
    # Antithetic sampling.
    do_anti = [False, True]
    for n in n_paths:
        h_matrix = np.zeros((n_bins, 2))
        grid = None
        for idx, anti in enumerate(do_anti):
            grid, h_matrix_tmp, mean, h_mean, h_std, r_error, r_error_path = \
                mc_simulation(instrument, price_a, hw, bond, instru_obj,
                              discount_curve, rng,
                              n_bins, price_min, price_max,
                              n_reps1, n_reps2, np.array([n]),
                              antithetic=anti)
            h_matrix[:, idx] = h_matrix_tmp[:, 0]
        p1 = plt.plot(grid, h_matrix[:, 0], "-b", label="Monte-Carlo")
        p2 = plt.plot(grid, h_matrix[:, 1], "-r", label="Antithetic")
        plots = p1 + p2
        plt.axvline(price_a, c="grey", ls=":")
        plt.xlabel("Price")
        plt.ylabel("Probability density")
        plt.legend(plots, [plot.get_label() for plot in plots])
        plt.savefig(f"density_2d_{instrument}_{n}.png")
        plt.clf()


@execution_time
def plot_type_3(instrument: str,
                price_a: float,
                hw: sde.SDE,
                bond: zero_coupon_bond.ZCBond,
                instru_obj,
                discount_curve: misc.DiscreteFunc,
                rng: np.random.Generator,
                n_bins: int,
                price_min: float,
                price_max: float,
                n_reps1: int,
                n_reps2: int,
                price_min_plot: float,
                price_max_plot: float):
    """
    Plot...

    Args:
        instrument: Name of instrument.
        price_a: Analytical price.
        hw: Hull-White SDE object.
        bond: Zero-coupon bond object.
        instru_obj: Instrument object.
        discount_curve: Discount curve represented on event-grid.
        rng: Random number generator.
        n_bins: Number of bins on price grid.
        price_min: Minimum value on price grid.
        price_max: Maximum value on price grid.
        n_reps1: Number of repetitions in outer for-loop.
        n_reps2: Number of repetitions in inner for-loop.
    """
    # Number of MC paths per simulation.
    n_mc = 8
    n_path_mul = 100
    path_grid = np.arange(1, n_mc + 1) * n_path_mul
    # Antithetic sampling.
    do_anti = [False, True]
    for anti in do_anti:
        grid, z_price, mean, h_mean, h_std, r_error, r_error_path = \
            mc_simulation(instrument, price_a, hw, bond, instru_obj,
                          discount_curve, rng,
                          n_bins, price_min, price_max,
                          n_reps1, n_reps2, path_grid,
                          antithetic=anti)
        x_price, y_price = np.meshgrid(path_grid, grid)
        plt.pcolormesh(x_price, y_price, z_price,
                       cmap=plt.colormaps['hot'], shading='gouraud')
        plt.xlabel("Number of Monte-Carlo paths")
        plt.ylabel("Price")
        plt.ylim(price_min_plot, price_max_plot)
        clb = plt.colorbar()  # plt.colorbar(ticks=np.arange(0, 36, 5))
        clb.set_label("Probability density")
        # plt.clim(0, 35)
        p1 = plt.plot(path_grid, h_mean, "-b", label="Mean")
        p2 = plt.plot(path_grid, h_mean + h_std, "ob", label="Mean +/- std")
        p3 = plt.plot(path_grid, h_mean - h_std, "ob")
        plots = p1 + p2
        plt.legend(plots, [plot.get_label() for plot in plots])
        plt.savefig(f"density_3d_{instrument}_{anti}.png")
        plt.clf()


@execution_time
def plot_type_4(instrument: str,
                price_a: float,
                hw: sde.SDE,
                bond: zero_coupon_bond.ZCBond,
                instru_obj,
                discount_curve: misc.DiscreteFunc,
                rng: np.random.Generator,
                n_reps: int,
                n_path_const: int):
    """
    Plot...

    Args:
        instrument: Name of instrument.
        price_a: Analytical price.
        hw: Hull-White SDE object.
        bond: Zero-coupon bond object.
        instru_obj: Instrument object.
        discount_curve: Discount curve represented on event-grid.
        rng: Random number generator.
        n_reps: Number of repetitions.
        n_path_const: ...
    """
    # Number of MC paths per simulation.
    n_paths = np.array([100 * 2 ** (2 * n) for n in range(n_path_const)])
    # Antithetic sampling.
    do_anti = [False, True]
    # Relative error, i.e. <| (price_n - price_a) / price_a |>.
    error_plot = np.zeros((len(do_anti), n_paths.size))
    # Relative error for one simulation.
    error_path_plot = np.zeros((len(do_anti), n_paths.size))
    for count1, anti in enumerate(do_anti):
        grid, h_matrix, mean, h_mean, h_std, r_error, r_error_path = \
            mc_simulation(instrument, price_a, hw, bond, instru_obj,
                          discount_curve, rng,
                          n_bins=10,
                          price_min=price_a - 0.1, price_max=price_a + 0.1,
                          n_reps1=1, n_reps2=n_reps,
                          n_paths_array=n_paths,
                          antithetic=anti)
        error_plot[count1, :] = r_error
        error_path_plot[count1, :] = r_error_path
    p1 = plt.plot(n_paths, error_plot[0, :], "ob", label="MC average")
    p2 = plt.plot(n_paths, error_path_plot[0, :], "-xb", label="MC estimates")
    p3 = plt.plot(n_paths, error_plot[1, :], "or", label="Antithetic average ")
    p4 = plt.plot(n_paths, error_path_plot[1, :],
                  "-xr", label="Antithetic estimates")
    plots = p1 + p2 + p3 + p4
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of paths")
    plt.ylabel("Relative error of Monet-Carlo estimate")
    plt.legend(plots, [plot.get_label() for plot in plots])
    plt.savefig(f"mc_error_{instrument}.png")
    plt.clf()


if __name__ == '__main__':

    start_time = time.perf_counter()

    # Speed of mean reversion strip.
    kappa = input.kappa_strip

    # Volatility strip.
    vol = input.vol_strip

    # Discount curve.
    discount_curve = input.disc_curve

    # Event grid.
    # For "real" test of Sobol sequence, include quarterly time steps...
    event_grid = np.arange(0, 31, 15)

    # Discount curve on event grid.
    discount_grid = discount_curve.interpolation(event_grid)
    discount_curve = misc.DiscreteFunc("discount", event_grid, discount_grid)

    # Integration step size.
    int_step_size = 1 / 52  # 1 / 365

    # SDE object.
    hw = sde.SDE(kappa, vol, event_grid, int_step_size)

    # Zero-coupon bond object. Maturing at last event.
    maturity_idx = event_grid.size - 1
    bond = zero_coupon_bond.ZCBond(kappa, vol, discount_curve, event_grid,
                                   maturity_idx, int_step_size)

    # European call option object.
    expiry_idx = 1
    maturity_idx = event_grid.size - 1
    strike = 0.7
    call = call_option.Call(kappa, vol, discount_curve, event_grid, strike,
                            expiry_idx, maturity_idx, int_step_size)

    # European put option object.
    expiry_idx = 1
    maturity_idx = event_grid.size - 1
    strike = 0.8
    put = put_option.Put(kappa, vol, discount_curve, event_grid, strike,
                         expiry_idx, maturity_idx, int_step_size)

    # Swap object.
    fixed_rate = 0.05
    swap_obj = swap.Swap(kappa, vol, discount_curve, event_grid,
                         fixed_rate, int_step_size)

    # Random number generator.
    rng = np.random.default_rng(0)

    # Instrument type.
#    instrument = "ZCBond"
#    instrument = "Call"
    instrument = "Put"
#    instrument = "Swap"

    # Number of price bins.
    n_bins = 200

    # Analytical price, and price-interval for histogram construction.
    event_idx = 0
    if instrument == "ZCBond":
        price_a = bond.price(0, event_idx)
        instru_obj = bond
        # Price interval.
        price_min = price_a - 0.15
        price_max = price_a + 0.15
        # Price interval for density_3d plot.
        price_min_plot = 0.44
        price_max_plot = 0.56
    elif instrument == "Call":
        price_a = call.price(0, event_idx)
        instru_obj = call
        # Price interval.
        price_min = price_a - 0.08
        price_max = price_a + 0.12
        # Price interval for density_3d plot.
        price_min_plot = 0.05
        price_max_plot = 0.15
    elif instrument == "Put":
        price_a = put.price(0, event_idx)
        instru_obj = put
        # Price interval.
        price_min = price_a - 0.03
        price_max = price_a + 0.03
        # Price interval for density_3d plot.
        price_min_plot = 0.085
        price_max_plot = 0.115
    elif instrument == "Swap":
        price_a = swap_obj.price(0, event_idx)
        instru_obj = swap_obj
        # Price interval.
        price_min = price_a - 0.25
        price_max = price_a + 0.2
        # Price interval for density_3d plot.
        price_min_plot = -0.5
        price_max_plot = -0.2
    else:
        price_a = 0
        instru_obj = bond
        # Price interval.
        price_min = price_a - 0.1
        price_max = price_a + 0.1
        # Price interval for density_3d plot.
        price_min_plot = price_min
        price_max_plot = price_max

    # Density plots type 1
    ######################
    # Number of repetitions of Monte-Carlo simulation.
    # 20 * 50_000, ~0.5 hours.
    n_reps1 = 20
    n_reps2 = 50_000

    plot_type_1(instrument, price_a, hw, bond, instru_obj,
                discount_curve, rng,
                n_bins, price_min, price_max, n_reps1, n_reps2)

    # Density plots type 2
    ######################
    # Number of repetitions of Monte-Carlo simulation.
    # 20 * 50_000, ~8 hours.
    n_reps1 = 20
    n_reps2 = 50_000

    plot_type_2(instrument, price_a, hw, bond, instru_obj,
                discount_curve, rng,
                n_bins, price_min, price_max, n_reps1, n_reps2)

    # Density plots type 3
    ######################
    # Number of repetitions of Monte-Carlo simulation.
    # 100 * 10_000, ~80 minutes.
    n_reps1 = 10
    n_reps2 = 10_000

    plot_type_3(instrument, price_a, hw, bond, instru_obj,
                discount_curve, rng,
                n_bins, price_min, price_max, n_reps1, n_reps2,
                price_min_plot, price_max_plot)

    # Convergence plots type 4
    ##########################
    # Number of repetitions of Monte-Carlo simulation.
    # 2000 + 8, ~60 minutes.
    n_reps = 2000
    n_path_const = 8

    plot_type_4(instrument, price_a, hw, bond, instru_obj,
                discount_curve, rng,
                n_reps, n_path_const)

    end_time = time.perf_counter()
    print("\nTotal execution time: ",
          datetime.timedelta(seconds=end_time - start_time))

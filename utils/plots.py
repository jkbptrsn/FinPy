import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from utils import global_types


def plot_price_and_greeks(instrument, show=True):
    """..."""

    plt.rcParams.update({"font.size": 10})

    grid = instrument.fd.grid

    # Figure 1
    f1, ax1 = plt.subplots(4, 1, sharex=True)
    f1.suptitle("Price and greeks of instrument")

    # Plot of instrument payoff and price.
    if instrument.model == global_types.Model.VASICEK and \
            instrument.type == global_types.Instrument.EUROPEAN_CALL:
        payoff = \
            instrument.payoff(instrument.zcbond.price(grid,
                                                      instrument.expiry_idx))
        ax1[0].plot(grid, payoff, '-.k', label="Payoff")
    else:
        ax1[0].plot(grid, instrument.payoff(grid), '-.k', label="Payoff")
    ax1[0].plot(grid, instrument.fd.solution, '-r', label="Numerical result")
    try:
        ax1[0].plot(grid, instrument.price(grid, 0),
                    'ob', markersize=2, label="Analytical result")
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 1, price")
    ax1[0].set_ylabel("Price")
    ax1[0].grid(True)
    ax1[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    ax1[0].legend(bbox_to_anchor=(0.1, 1.02, 0.9, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3)

    # Plot of instrument delta.
    ax1[1].plot(grid, instrument.fd.delta(), '-r')
    try:
        ax1[1].plot(grid, instrument.delta(grid, 0), 'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 1, delta")
    ax1[1].set_ylabel("Delta")
    ax1[1].grid(True)
    ax1[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Plot of instrument gamma.
    ax1[2].plot(grid, instrument.fd.gamma(), '-r')
    try:
        ax1[2].plot(grid, instrument.gamma(grid, 0), 'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 1, gamma")
    ax1[2].set_ylabel("Gamma")
    ax1[2].grid(True)
    ax1[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Plot of instrument theta.
    ax1[3].plot(grid, instrument.fd.theta(), '-r')
    try:
        ax1[3].plot(grid, instrument.theta(grid, 0), 'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 1, theta")
    ax1[3].set_ylabel("Theta")
    ax1[3].set_xlabel("\"Value\" of underlying")
    ax1[3].grid(True)
    ax1[3].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Figure 2
    f2, ax2 = plt.subplots(4, 1, sharex=True)
    f2.suptitle("Analytical result minus numerical result")

    # Plot of instrument price.
    try:
        ax2[0].plot(grid, instrument.price(grid, 0) - instrument.fd.solution,
                    'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 2, price")
    ax2[0].set_ylabel("Price")
    ax2[0].grid(True)
    ax2[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Plot of instrument delta.
    try:
        ax2[1].plot(grid, instrument.delta(grid, 0) - instrument.fd.delta(),
                    'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 2, delta")
    ax2[1].set_ylabel("Delta")
    ax2[1].grid(True)
    ax2[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Plot of instrument gamma.
    try:
        ax2[2].plot(grid, instrument.gamma(grid, 0) - instrument.fd.gamma(),
                    'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 2, gamma")
    ax2[2].set_ylabel("Gamma")
    ax2[2].grid(True)
    ax2[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Plot of instrument theta.
    try:
        ax2[3].plot(grid, instrument.theta(grid, 0) - instrument.fd.theta(),
                    'ob', markersize=2)
    except (AttributeError, TypeError, ValueError):
        print("Error in plots.py: Fig 2, theta")
    ax2[3].set_ylabel("Theta")
    ax2[3].set_xlabel("\"Value\" of underlying")
    ax2[3].grid(True)
    ax2[3].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    if show:
        plt.show()


def plot_path(time_grid, path, show=True):
    """..."""

    f1, ax1 = plt.subplots(2, 1, sharex=True)
    f1.suptitle("Monte-Carlo scenario")

    ax1[0].plot(time_grid, path[0], 'b')
    ax1[0].grid(True)
    ax1[0].set_ylabel("Stochastic process")

    if len(path) == 2:
        ax1[1].plot(time_grid, path[1], 'b')
        ax1[1].grid(True)
        ax1[1].set_ylabel("Discount curve")
        ax1[1].set_xlabel("Time")
    else:
        ax1[0].set_xlabel("Time")

    if show:
        plt.show()


def plot_rate_distribution(event_idx, rate, mean, std):
    """..."""
    n_bins = 101
    r_min = rate[event_idx, :].min()
    r_max = rate[event_idx, :].max()
    r_interval = r_max if r_max > abs(r_min) else abs(r_min)
    bins = np.arange(n_bins) * 2 * r_interval / (n_bins - 1) - r_interval
    plt.hist(rate[event_idx, :], bins=bins, density=True)

    grid = (bins[1:] + bins[:-1]) / 2
    plt.plot(grid, norm.pdf(grid, loc=mean, scale=std))
    print(mean, std)

    plt.show()
#    plt.pause(2.2)


def plot_rate_discount_distribution(event_idx, rate, discount):
    """..."""
#    n_bins = 101
#    r_min = rate[event_idx, :].min()
#    r_max = rate[event_idx, :].max()
#    r_interval = r_max if r_max > abs(r_min) else abs(r_min)
#    bins = np.arange(n_bins) * 2 * r_interval / (n_bins - 1) - r_interval

    bin_range = [[-0.15, 0.15], [0, 3]]

    plt.hist2d(rate[event_idx, :], discount[event_idx, :],
               bins=100, range=bin_range, density=True)

#    plt.show()
    plt.pause(0.5)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot1(solver, payoff, price, instrument=None, show=True):
    """..."""

    plt.rcParams.update({"font.size": 10})

    # Figure 1
    f1, ax1 = plt.subplots(4, 1, sharex=True)
    f1.suptitle("Price and greeks of instrument")

    # Price
    ax1[0].plot(solver.grid(), payoff, 'k')
    ax1[0].plot(solver.grid(), price, 'r')
    if instrument:
        try:
            ax1[0].plot(solver.grid(), instrument.price(solver.grid(), 0), 'ob', markersize=3)
        except AttributeError:
            pass
    ax1[0].set_ylabel("Price")
    ax1[0].grid(True)

    # Delta
    ax1[1].plot(solver.grid(), solver.delta_fd(), 'r')
    if instrument:
        try:
            ax1[1].plot(solver.grid(), instrument.delta(solver.grid(), 0), 'ob', markersize=3)
        except (AttributeError, ValueError):
            print("Error in plots.py")
    ax1[1].set_ylabel("Delta")
    ax1[1].grid(True)

    # Gamma
    ax1[2].plot(solver.grid(), solver.gamma_fd(), 'r')
    if instrument:
        try:
            ax1[2].plot(solver.grid(), instrument.gamma(solver.grid(), 0), 'ob', markersize=3)
        except (AttributeError, ValueError):
            print("Error in plots.py")
    ax1[2].set_ylabel("Gamma")
    ax1[2].grid(True)

    # Theta
    ax1[3].plot(solver.grid(), solver.theta_fd(), 'r')
    if instrument:
        try:
            ax1[3].plot(solver.grid(), instrument.theta(solver.grid(), 0), 'ob', markersize=3)
        except (AttributeError, ValueError):
            print("Error in plots.py")
    ax1[3].set_ylabel("Theta")
    ax1[3].set_xlabel("\"Value\" of underlying")
    ax1[3].grid(True)

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

#    plt.show()
    plt.pause(0.2)


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

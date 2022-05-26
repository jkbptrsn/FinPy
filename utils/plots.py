import matplotlib.pyplot as plt


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

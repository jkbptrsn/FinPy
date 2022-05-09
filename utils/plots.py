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
    ax1[3].set_xlabel("\"Price\" of underlying")
    ax1[3].grid(True)

    if show:
        plt.show()

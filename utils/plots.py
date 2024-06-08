import matplotlib.pyplot as plt


def plot_price_and_greeks(
        instrument,
        show: bool = True) -> None:
    """Plot price, delta, gamma and theta on finite difference grid."""
    # Finite difference grid.
    grid = instrument.fd.grid

    plt.rcParams.update({"font.size": 10})

    # Figure 1
    f1, ax1 = plt.subplots(4, 1, sharex="all")
    f1.suptitle("Price and greeks of instrument")

    # Plot of instrument payoff and price.
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
    f2, ax2 = plt.subplots(4, 1, sharex="all")
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

import matplotlib.pyplot as plt


def plot1(solver, payoff, price, dt, show=True):

    plt.rcParams.update({'font.size': 10})

    # Figure 1
    f1, ax1 = plt.subplots(4, 1, sharex=True)
    ax1[0].plot(solver.grid(), payoff, 'k')
    ax1[0].set_ylabel("Price")
    ax1[0].grid(True)

    # Price function
    ax1[0].plot(solver.grid(), price, 'r')

    # Delta
    ax1[1].plot(solver.grid(), solver.fd_delta(solver.grid(), price), 'r')
    ax1[1].set_ylabel("Delta")
    ax1[1].grid(True)

    # Gamma
    ax1[2].plot(solver.grid(), solver.fd_gamma(solver.grid(), price), 'r')
    ax1[2].set_ylabel("Gamma")
    ax1[2].grid(True)

    # Theta
    # Let dt be an attribute of solver class
    ax1[3].plot(solver.grid(), solver.fd_theta(dt, price), 'r')
    ax1[3].set_ylabel("Theta")
    ax1[3].set_xlabel("\"Value\" of underlying")
    ax1[3].grid(True)

    if show:
        plt.show()

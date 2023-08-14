from matplotlib import pyplot as plt
import numpy as np
import numpy.polynomial as poly


def american_option(instrument,
                    basis_set: str = "Power",
                    degree: int = 4):
    """Pricing American option using Longstaff-Schwartz method.

    Args:
        instrument: ...
        basis_set: Type of polynomial basis set. Default is Power.
            * Power
            * Chebyshev
            * Legendre
            * Laguerre
            * Hermite
        degree: Degree of series based on basis set.

    Returns:
        American option price.
    """
    paths = instrument.mc_exact.solution
    discount_grid = instrument.mc_exact.discount_grid
    exercise_grid = instrument.exercise_grid

    plot_regression = False

    # Get fitting function.
    if basis_set == "Power":
        fit_function = poly.Polynomial.fit
    elif basis_set == "Chebyshev":
        fit_function = poly.Chebyshev.fit
    elif basis_set == "Legendre":
        fit_function = poly.Legendre.fit
    elif basis_set == "Laguerre":
        fit_function = poly.Laguerre.fit
    elif basis_set == "Hermite":
        fit_function = poly.Hermite.fit
    else:
        raise ValueError(f"Type of basis set is unknown: {basis_set}")
    # Index of exercise (on event grid) for each path.
    exercise_index = -np.ones(paths.shape[1], dtype=int)
    # Continuation value for each path.
    cont_value_path = np.zeros(paths.shape[1])

    # Loop over exercise indices (in reverse order).
    # TODO: Do the "reverse" here.
    idx_old = None
    first_exercise = True
    for idx in exercise_grid:
        # Continuation value discounted to current exercise event.
        if idx_old is None:
            discount_factor = 1
            idx_old = idx
        else:
            discount_factor = discount_grid[idx_old] / discount_grid[idx]
            idx_old = idx
        cont_value_path *= discount_factor
        # Immediate exercise value.
        exercise_value = instrument.payoff(paths[idx, :])
        # Least squares fit.
        ls_fit = fit_function(paths[idx, :], cont_value_path, deg=degree)
        # LSM estimate of expected continuation value.
        cont_value_lsm = ls_fit(paths[idx, :])

        # The polynomial fit can give negative continuation values.
        # These are truncated at zero. TODO: Important!
        cont_value_lsm = np.maximum(cont_value_lsm, 0)

        # Exercise if exercise value is large than continuation value.
        if first_exercise:
            do_exercise = np.maximum(exercise_value, 0)
            first_exercise = False
        else:
            do_exercise = np.maximum(exercise_value - cont_value_lsm, 0)
        do_exercise_idx = np.nonzero(do_exercise)
        # Update exercise indices.
        exercise_index[do_exercise_idx] = idx
        # Update continuation values.
        cont_value_path[do_exercise_idx] = exercise_value[do_exercise_idx]

        if plot_regression:
            x_grid = np.linspace(10, 120)
            plt.plot(paths[idx, :], cont_value_path, "ob")
            plt.plot(x_grid, ls_fit(x_grid), "-r")
            plt.plot(x_grid, np.maximum(ls_fit(x_grid), 0), "-k")
            plt.pause(0.5)
            plt.cla()

    # Monte-Carlo average.
    mc_average = cont_value_path.sum() / cont_value_path.size
    # Possible final discounting back to time zero.
    mc_average *= discount_grid[exercise_grid[-1]] / discount_grid[0]

    return mc_average


def prepayment_option(paths,
                      bond_payoff,
                      strike_price: float = 100,
                      basis_set: str = "Power",
                      degree: int = 4):
    """Pricing prepayment option using Longstaff-Schwartz method.

    Args:
        paths: ...
        bond_payoff: ...
        strike_price: ...
        basis_set: Type of polynomial basis set. Default is Power.
            * Power
            * Chebyshev
            * Legendre
            * Laguerre
            * Hermite
        degree: Degree of series based on basis set. Default is 4.

    Returns:
        Prepayment option price.
    """
    plot_regression = False

    # Get fitting function.
    if basis_set == "Power":
        fit_function = poly.Polynomial.fit
    elif basis_set == "Chebyshev":
        fit_function = poly.Chebyshev.fit
    elif basis_set == "Legendre":
        fit_function = poly.Legendre.fit
    elif basis_set == "Laguerre":
        fit_function = poly.Laguerre.fit
    elif basis_set == "Hermite":
        fit_function = poly.Hermite.fit
    else:
        raise ValueError(f"Unknown basis set: {basis_set}")

    # Least squares fit.
    ls_fit = fit_function(paths, bond_payoff, deg=degree)
    # LSM estimate of expected continuation value.
    cont_value_lsm = ls_fit(paths)

    # The polynomial fit can give negative continuation values.
    # These are truncated at zero. TODO: Important!
    cont_value_lsm = np.maximum(cont_value_lsm, 0)

    if plot_regression:
        x_grid = np.linspace(-0.25, 0.15)
        plt.plot(paths, bond_payoff, "ob")
        plt.plot(x_grid, ls_fit(x_grid), "-r")
        plt.plot(x_grid, np.maximum(ls_fit(x_grid), 0), "-k")
        plt.pause(0.5)
        plt.cla()

    return np.maximum(cont_value_lsm - strike_price, 0)

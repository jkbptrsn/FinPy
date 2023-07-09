from matplotlib import pyplot as plt
import numpy as np
import numpy.polynomial as poly


def american_option(instrument,
                    exercise_grid: np.ndarray,
                    basis_set: str = "Power",
                    degree: int = 3):
    """Pricing American option using Longstaff-Schwartz method.

    Args:
        instrument: ...
        exercise_grid: Exercise event indices (in reverse order) on
            event grid.
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
    cont_value = np.zeros(paths.shape[1])

    # Loop over exercise indices (in reverse order).
    # TODO: Do the "reverse" here.
    idx_old = None
    for idx in exercise_grid:
        # Continuation value discounted to current exercise event.
        if idx_old is None:
            discount_factor = 1
            idx_old = idx
        else:
            discount_factor = discount_grid[idx_old] / discount_grid[idx]
            idx_old = idx
        cont_value *= discount_factor
        # Immediate exercise value.
        exercise_value = instrument.payoff(paths[idx, :])
        # Least squares fit.
        ls_fit = fit_function(paths[idx, :], cont_value, deg=degree)
        # LSM estimate of continuation value.
        lsm_cont_value = ls_fit(paths[idx, :])

        # The polynomial fit can give negative continuation values.
        # These are truncated at zero. TODO: Important!
        lsm_cont_value = np.maximum(lsm_cont_value, 0)

        # Exercise if exercise value is large than continuation value.
        do_exercise = np.maximum(exercise_value - lsm_cont_value, 0)
        do_exercise_idx = np.nonzero(do_exercise)
        # Update exercise indices.
        exercise_index[do_exercise_idx] = idx
        # Update continuation values.
        cont_value[do_exercise_idx] = exercise_value[do_exercise_idx]

        if plot_regression:
            x_grid = np.linspace(10, 120)
            plt.plot(paths[idx, :], cont_value, "ob")
            plt.plot(x_grid, ls_fit(x_grid), "-r")
            plt.plot(x_grid, np.maximum(ls_fit(x_grid), 0), "-k")
            plt.pause(0.5)
            plt.cla()

    ##################
    # TODO: make exercise_grid and attribute of instrument object.

    mc_average = 0
    for m in range(paths.shape[1]):
        idx = exercise_index[m]
        if idx != -1:

#            exercise_value = np.max(strike - paths[idx, m], 0)
            exercise_value = instrument.payoff(paths[idx, m])
            mc_average += exercise_value * discount_grid[idx]

    mc_average /= paths.shape[1]

    if False:
        plt.hist(exercise_index)
        plt.show()

    return mc_average

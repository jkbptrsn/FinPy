import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial as poly

plot_regression = False


def black_scholes(
        instrument,
        basis_set: str = "Power",
        degree: int = 4) -> (float, float):
    """Pricing American option in Black-Scholes model.

    Args:
        instrument: Financial instrument object.
        basis_set: Type of polynomial basis set. Default is "Power".
            - "Power"
            - "Chebyshev"
            - "Legendre"
            - "Laguerre"
            - "Hermite"
        degree: Degree of basis set series. Default is 4.

    Returns:
        American option price.
    """
    paths = instrument.mc_exact.price_paths
    discount_grid = instrument.mc_exact.discount_grid
    exercise_grid = instrument.exercise_grid

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

    # Index of exercise (on event grid) for each path.
    exercise_idx = -np.ones(paths.shape[1], dtype=int)
    # Discounted payoff along each path.
    payoff = np.zeros(paths.shape[1])

    # Loop over exercise indices (in reverse order).
    idx_old = None
    first_exercise = True
    ls_fit = None
    for idx in np.flip(exercise_grid):
        # Payoff, along each path, discounted to current event.
        if first_exercise:
            discount_factor = 1
            idx_old = idx
        else:
            discount_factor = discount_grid[idx_old] / discount_grid[idx]
            idx_old = idx
        payoff *= discount_factor
        # Option payoff, for each path, if exercised at current event.
        exercise_value = instrument.payoff(paths[idx, :])
        # Indices of paths for which the option is ITM.
        path_idx_itm = np.nonzero(exercise_value)[0]

        # Option is exercised if 'exercise value' is larger than
        # 'continuation value'.
        if first_exercise:
            path_idx_exercise = path_idx_itm
            first_exercise = False
        else:
            # Try to fit based on paths for which the option is ITM.
            try:
                # Least squares fit.
                paths_itm = paths[idx, path_idx_itm]
                ls_fit = (
                    fit_function(paths_itm, payoff[path_idx_itm], deg=degree))
                # LSM estimate of expected continuation value.
                cont_value_lsm = ls_fit(paths_itm)
                # Least squares fit might result in negative values!
                cont_value_lsm = np.maximum(cont_value_lsm, 0)
                exercise = np.maximum(exercise_value[path_idx_itm]
                                      - cont_value_lsm, 0)
                path_idx_exercise = path_idx_itm[np.nonzero(exercise)[0]]
            except:
                # Least squares fit.
                ls_fit = fit_function(paths[idx, :], payoff, deg=degree)
                # LSM estimate of expected continuation value.
                cont_value_lsm = ls_fit(paths[idx, :])
                # Least squares fit might result in negative values!
                cont_value_lsm = np.maximum(cont_value_lsm, 0)
                exercise = np.maximum(exercise_value - cont_value_lsm, 0)
                path_idx_exercise = np.nonzero(exercise)

        # Update index of exercise for each path.
        exercise_idx[path_idx_exercise] = idx
        # Update discounted payoff along each path.
        payoff[path_idx_exercise] = exercise_value[path_idx_exercise]

        if plot_regression:
            x_grid = np.linspace(20, 50)
            plt.plot(paths[idx, :], payoff, "ob", label="MC paths")
            if ls_fit:
                plt.plot(x_grid, ls_fit(x_grid), "-r", label="LS fit")
                plt.plot(x_grid, np.maximum(ls_fit(x_grid), 0),
                         "-k", label="LS fit floored")
            plt.xlabel("Stock price")
            plt.ylabel("Option price")
            plt.legend()
            plt.pause(0.5)
            plt.cla()

    # Possible final discounting back to time zero.
    payoff *= discount_grid[exercise_grid[0]] / discount_grid[0]
    mc_estimate = payoff.mean()
    mc_error = payoff.std(ddof=1)
    mc_error /= math.sqrt(paths.shape[1])

    return mc_estimate, mc_error


def prepayment_option(paths,
                      bond_payoff,
                      strike_price: float = 100,
                      basis_set: str = "Power",
                      degree: int = 4):
    """Pricing prepayment option using Longstaff-Schwartz method.

    Args:
        paths: Pseudo short rates.
        bond_payoff: Payoff along each path.
        strike_price: Strike price of bond.
        basis_set: Type of polynomial basis set. Default is "Power".
            - "Power"
            - "Chebyshev"
            - "Legendre"
            - "Laguerre"
            - "Hermite"
        degree: Degree of series based on basis set. Default is 4.

    Returns:
        Prepayment option price.
    """
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

    # TODO: Only include in-the-money paths???

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

import numpy as np
import scipy


def normal_realizations(
        n_realizations: int,
        rng: np.random.Generator,
        antithetic: bool = False) -> np.ndarray:
    """Realizations of standard normal random variable.

    Args:
        n_realizations: Number of realizations.
        rng: Random number generator.
        antithetic: Antithetic sampling for variance reduction.
            Default is False.

    Returns:
        Realizations.
    """
    if antithetic and n_realizations % 2 == 1:
        raise ValueError("In antithetic sampling, the number of "
                         "realizations should be even.")
    anti = 1
    if antithetic:
        anti = 2
    realizations = rng.standard_normal(size=n_realizations // anti)
    if antithetic:
        realizations = np.append(realizations, -realizations)
    return realizations


def cholesky_2d(
        correlation: float,
        n_sets: int,
        rng: np.random.Generator,
        antithetic: bool = False) -> (np.ndarray, np.ndarray):
    """Cholesky decomposition of correlation matrix in 2-D.

    In 2-D, the transformation is simply:
        (x1, correlation * x1 + sqrt{1 - correlation ** 2} * x2).

    Args:
        correlation: Correlation scalar.
        n_sets: Number of realization of two correlated standard normal
            random variables.
        rng: Random number generator.
        antithetic: Antithetic sampling for variance reduction.
            Default is False.

    Returns:
        Realizations of two correlated standard normal random variables.
    """
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    corr_matrix = np.linalg.cholesky(corr_matrix)
    x1 = normal_realizations(n_sets, rng, antithetic)
    x2 = normal_realizations(n_sets, rng, antithetic)
    return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
        corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2


def trapz(
        grid: np.ndarray,
        function: np.ndarray) -> np.ndarray:
    """Trapezoidal integration.

    Args:
        grid: Grid on which the function is represented.
        function: Function value for each grid point.

    Returns:
        Trapezoidal integral for each step along the grid.
    """
    dx = np.diff(grid)
    return dx * (function[1:] + function[:-1]) / 2


########################################################################


def sobol_init(event_grid_size: int):
    """Initialization of sobol sequence generator for two 1-dimensional
    processes for event_grid_size time steps..."""
    # Sobol sequence generator for seq_size = event_grid_size - 1.
    seq_size = event_grid_size - 1
    return scipy.stats.qmc.Sobol(2 * seq_size)


def cholesky_2d_sobol_test(
        correlation: float,
        sobol_norm: np.ndarray,
        time_idx: int) -> (np.ndarray, np.ndarray):
    """..."""
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    corr_matrix = np.linalg.cholesky(corr_matrix)
    x1 = normal_realizations_sobol_test(sobol_norm[:, 2 * (time_idx - 1)])
    x2 = normal_realizations_sobol_test(sobol_norm[:, 2 * (time_idx - 1) + 1])
    return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
        corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2


def normal_realizations_sobol_test(sobol_norm: np.ndarray) -> np.ndarray:
    """..."""
    return sobol_norm

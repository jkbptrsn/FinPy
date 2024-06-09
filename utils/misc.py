import numpy as np
from scipy.stats import norm, qmc


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


def sobol_generator(
        mc_dimension: int,
        event_grid_size: int,
        seed: int = None) -> qmc.Sobol:
    """Initialization of Sobol sequence generator.

    Args:
        mc_dimension: Number of random numbers per event date.
        event_grid_size: Number of event dates.
        seed: Seed of

    Returns:
        Sobol sequence generator.
    """
    sequence_size = mc_dimension * (event_grid_size - 1)
    return qmc.Sobol(sequence_size, seed=seed)


def sobol_sequence(
        n_paths: int,
        generator: qmc.Sobol) -> np.ndarray:
    """Construct Sobol sequence.

    TODO: Use random_base2 instead? Check n_paths = 2 ** integer...

    Args:
        n_paths: Number of Monte-Carlo paths.
        generator: Sobol sequence generator.

    Returns:
        Sobol sequence.
    """
    return generator.random(n_paths)


def cholesky_2d_sobol(
        correlation: float,
        sobol_seq: np.ndarray,
        event_idx: int) -> (np.ndarray, np.ndarray):
    """Cholesky decomposition of correlation matrix in 2-D.

    Args:
        correlation: Correlation scalar.
        sobol_seq:
        event_idx: Index on event grid.

    Returns:
        Realizations of two correlated standard normal random variables.
    """
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    corr_matrix = np.linalg.cholesky(corr_matrix)
    x1 = norm.ppf(sobol_seq[:, 2 * (event_idx - 1)])
    x2 = norm.ppf(sobol_seq[:, 2 * (event_idx - 1) + 1])
    return corr_matrix[0][0] * x1 + corr_matrix[0][1] * x2, \
        corr_matrix[1][0] * x1 + corr_matrix[1][1] * x2

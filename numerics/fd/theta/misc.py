import numpy as np

from numerics.fd.theta import linear_algebra as la
from numerics.fd.theta import differential_operators as do

info = """
    Tridiagonal form:
        - 1st row: Superdiagonal (exclude first element)
        - 2nd row: Diagonal
        - 3rd row: Subdiagonal (exclude last element)

    Pentadiagonal form:
        - 1st row: 2nd superdiagonal (exclude first two elements)
        - 2nd row: 1st superdiagonal (exclude first element)
        - 3rd row: Diagonal
        - 4th row: 1st subdiagonal (exclude last element)
        - 5th row: 2nd subdiagonal (exclude last two elements)
"""


def delta_equidistant(dx: float,
                      function: np.ndarray,
                      form: str = "tri") -> np.ndarray:
    """Finite difference calculation of delta on equidistant grid.

    Assuming ascending grid.

    Args:
        dx: Equidistant spacing.
        function: ...
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Delta
    """
    operator = do.ddx_equidistant(function.size, dx, form)
    return la.matrix_col_prod(operator, function, form)


def delta(grid: np.ndarray,
          function: np.ndarray,
          form: str = "tri") -> np.ndarray:
    """Finite difference calculation of delta on non-equidistant grid.

    Assuming ascending grid.

    Args:
        grid: ...
        function: ...
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Delta
    """
    operator = do.ddx(grid, form)
    return la.matrix_col_prod(operator, function, form)


def gamma_equidistant(dx: float,
                      vector: np.ndarray,
                      form: str = "tri") -> np.ndarray:
    """Finite difference calculation of gamma on equidistant grid.

    Assuming ascending grid.

    Args:
        dx: Equidistant spacing.
        vector: ...
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Gamma
    """
    operator = do.d2dx2_equidistant(vector.size, dx, form)
    return la.matrix_col_prod(operator, vector, form)


def gamma(grid: np.ndarray,
          function: np.ndarray,
          form: str = "tri") -> np.ndarray:
    """Finite difference calculation of gamma on non-equidistant grid.

    Assuming ascending grid.

    Args:
        grid: ...
        function: ...
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Gamma
    """
    operator = do.d2dx2(grid, form)
    return la.matrix_col_prod(operator, function, form)


def norms_1d(vector1: np.ndarray,
             vector2: np.ndarray,
             step_size1: float,
             slice_nr=2):
    """...

    Args:
        vector1: Solution on grid1.
        vector2: Solution on grid2.
        step_size1: Step size of grid1.
        slice_nr: Ratio of number of steps of grid1 and grid2.

    Returns:
        Center norm, max norm, L2 norm
    """
    # Absolute difference.
    diff = np.abs(vector1 - vector2[::slice_nr])
    # "Center" norm.
    n_states = diff.size
    idx_center = (n_states - 1) // 2
    norm_center = diff[idx_center]
    # Max norm.
    norm_max = np.amax(diff)
    # L2 norm.
    norm_l2 = np.sqrt(np.sum(np.square(diff)) * step_size1)
    return norm_center, norm_max, norm_l2

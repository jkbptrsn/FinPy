import numpy as np

from numerical_methods.finite_difference.theta import linear_algebra as linalg
from numerical_methods.finite_difference.theta import differential_operators as do

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
    return linalg.matrix_col_prod(operator, function, form)


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
    return linalg.matrix_col_prod(operator, function, form)


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
    return linalg.matrix_col_prod(operator, vector, form)


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
    return linalg.matrix_col_prod(operator, function, form)

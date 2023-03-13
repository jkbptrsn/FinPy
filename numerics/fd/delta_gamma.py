import numpy as np

from numerics.fd import differential_operators as do
from numerics.fd import linear_algebra as la

info = """Banded matrix as 2-dimensional numpy array.

    Structure of numpy array consistent matrix diagonal ordered form 
    used in scipy.linalg.solve_banded function.

    Tridiagonal form:
        - 1st row: Superdiagonal (exclude first element).
        - 2nd row: Main diagonal.
        - 3rd row: Subdiagonal (exclude last element).

    Pentadiagonal form:
        - 1st row: 2nd superdiagonal (exclude first two elements).
        - 2nd row: 1st superdiagonal (exclude first element).
        - 3rd row: Main diagonal.
        - 4th row: 1st subdiagonal (exclude last element).
        - 5th row: 2nd subdiagonal (exclude last two elements).
"""


def delta_equidistant(dx: float,
                      function: np.ndarray,
                      band: str = "tri") -> np.ndarray:
    """Finite difference approximation of delta on equidistant grid.

    Assuming ascending grid.

    Args:
        dx: Constant grid spacing.
        function: Function values on grid.
        band: Tri- ("tri") or pentadiagonal ("penta") form. Default is
            tridiagonal.

    Returns:
        Delta.
    """
    operator = do.ddx_equidistant(function.size, dx, band)
    return la.matrix_col_prod(operator, function, band)


def delta(grid: np.ndarray,
          function: np.ndarray,
          band: str = "tri") -> np.ndarray:
    """Finite difference approximation of delta on non-equidistant grid.

    Assuming ascending grid.

    Args:
        grid: Non-equidistant grid.
        function: Function values on grid.
        band: Tri- ("tri") or pentadiagonal ("penta") form. Default is
            tridiagonal.

    Returns:
        Delta.
    """
    operator = do.ddx(grid, band)
    return la.matrix_col_prod(operator, function, band)


def gamma_equidistant(dx: float,
                      function: np.ndarray,
                      band: str = "tri") -> np.ndarray:
    """Finite difference approximation of gamma on equidistant grid.

    Assuming ascending grid.

    Args:
        dx: Constant grid spacing.
        function: Function values on grid.
        band: Tri- ("tri") or pentadiagonal ("penta") form. Default is
            tridiagonal.

    Returns:
        Gamma.
    """
    operator = do.d2dx2_equidistant(function.size, dx, band)
    return la.matrix_col_prod(operator, function, band)


def gamma(grid: np.ndarray,
          function: np.ndarray,
          band: str = "tri") -> np.ndarray:
    """Finite difference approximation of gamma on non-equidistant grid.

    Assuming ascending grid.

    Args:
        grid: Non-equidistant grid.
        function: Function values on grid.
        band: Tri- ("tri") or pentadiagonal ("penta") form. Default is
            tridiagonal.

    Returns:
        Gamma.
    """
    operator = do.d2dx2(grid, band)
    return la.matrix_col_prod(operator, function, band)

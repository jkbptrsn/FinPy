import numpy as np

from numerics.fd import linear_algebra as la


info = """
    Banded matrix as 2-dimensional numpy array.

    Structure of numpy array consistent with "matrix diagonal ordered 
    form" used in scipy.linalg.solve_banded function.

    Tridiagonal form:
        - 1st row: Superdiagonal (excl. first element).
        - 2nd row: Main diagonal.
        - 3rd row: Subdiagonal (excl. last element).

    Pentadiagonal form:
        - 1st row: 2nd superdiagonal (excl. first two elements).
        - 2nd row: 1st superdiagonal (excl. first element).
        - 3rd row: Main diagonal.
        - 4th row: 1st subdiagonal (excl. last element).
        - 5th row: 2nd subdiagonal (excl. last two elements).
"""


def ddx_equidistant(size: int,
                    dx: float,
                    band: str = "tri") -> np.ndarray:
    """FD approximation of 1st order differential operator.

    Central finite difference approximation of 1st order differential
    operator on equidistant grid. At the boundaries, 1st order
    forward/backward difference is used. Assuming ascending grid.

    Args:
        size: Number of elements along main diagonal.
        dx: Constant grid spacing.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Discrete 1st order differential operator.
    """
    if band == "tri":
        matrix = np.zeros((3, size))
        # 1st order forward difference at lower boundary.
        matrix[0, 1] = 1
        matrix[1, 0] = -1
        # 2nd order central difference.
        matrix[0, 2:] = 1 / 2
        matrix[2, :-2] = -1 / 2
        # 1st order backward difference at upper boundary.
        matrix[1, -1] = 1
        matrix[2, -2] = -1
    elif band == "penta":
        matrix = np.zeros((5, size))
        # 1st order forward difference at lower boundary.
        matrix[1, 1] = 1
        matrix[2, 0] = -1
        # 2nd order central difference at next-to lower boundary.
        matrix[1, 2] = 1 / 2
        matrix[3, 0] = -1 / 2
        # 4th order central difference.
        matrix[0, 4:] = -1 / 12
        matrix[1, 3:] = 2 / 3
        matrix[3, :-3] = -2 / 3
        matrix[4, :-4] = 1 / 12
        # 2nd order central difference at next-to upper boundary.
        matrix[1, -1] = 1 / 2
        matrix[3, -3] = -1 / 2
        # 1st order backward difference at upper boundary.
        matrix[2, -1] = 1
        matrix[3, -2] = -1
    else:
        raise ValueError(f"{band}: Unknown banded matrix. Use tri or penta.")
    return matrix / dx


def ddx(grid: np.ndarray,
        band: str = "tri") -> np.ndarray:
    """FD approximation of 1st order differential operator.

    Finite difference approximation of 1st order differential operator
    on non-equidistant grid. At the boundaries, 1st order
    forward/backward difference is used. Assuming ascending grid.

    See Sundqvist & Veronis (1970).

    Args:
        grid: Grid points.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Discrete 1st order differential operator.
    """
    if band == "tri":
        matrix = np.zeros((3, grid.size))
        # 1st order forward difference at lower boundary.
        dx = grid[1] - grid[0]
        matrix[0, 1] = 1 / dx
        matrix[1, 0] = -1 / dx
        # "2nd" order central difference.
        dx_p1 = grid[2:] - grid[1:-1]
        dx_m1 = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_p1 * (1 + dx_p1 / dx_m1))
        matrix[0, 2:] = factor
        matrix[1, 1:-1] = (np.square(dx_p1 / dx_m1) - 1) * factor
        matrix[2, :-2] = -np.square(dx_p1 / dx_m1) * factor
        # 1st order backward difference at upper boundary.
        dx = grid[-1] - grid[-2]
        matrix[1, -1] = 1 / dx
        matrix[2, -2] = -1 / dx
    elif band == "penta":
        matrix = np.zeros((5, grid.size))
        # 1st order forward difference at lower boundary.
        dx = grid[1] - grid[0]
        matrix[1, 1] = 1 / dx
        matrix[2, 0] = -1 / dx
        # "2nd" order central difference at next-to lower boundary.
        dx_p1 = grid[2] - grid[1]
        dx_m1 = grid[1] - grid[0]
        factor = 1 / (dx_p1 * (1 + dx_p1 / dx_m1))
        matrix[1, 2] = factor
        matrix[2, 1] = (np.square(dx_p1 / dx_m1) - 1) * factor
        matrix[3, 0] = -np.square(dx_p1 / dx_m1) * factor
        # "4th" order central difference.
        dx_p2 = grid[4:] - grid[3:-1]
        dx_p1 = grid[3:-1] - grid[2:-2]
        dx_m1 = grid[2:-2] - grid[1:-3]
        dx_m2 = grid[1:-3] - grid[:-4]
        factor = ((dx_m1 + dx_m2) ** 2 * (dx_p1 + dx_p2)
                  - 32 * dx_p1 * dx_m1 ** 2 - 32 * dx_p1 ** 2 * dx_m1
                  + (dx_m1 + dx_m2) * (dx_p1 + dx_p2) ** 2)
        matrix[0, 4:] = np.square(dx_m1 + dx_m2) * factor
        matrix[1, 3:-1] = -32 * np.square(dx_m1) * factor
        matrix[2, 2:-2] = -(np.square(dx_m1 + dx_m2)
                            - 32 * np.square(dx_m1) + 32 * np.square(dx_p1)
                            - np.square(dx_p1 + dx_p2)) * factor
        matrix[3, 1:-3] = 32 * np.square(dx_p1) * factor
        matrix[4, :-4] = -np.square(dx_p1 + dx_p2) * factor
        # 2nd order central difference at next-to upper boundary.
        dx_p1 = grid[-1] - grid[-2]
        dx_m1 = grid[-2] - grid[-3]
        factor = 1 / (dx_p1 * (1 + dx_p1 / dx_m1))
        matrix[1, -1] = factor
        matrix[2, -2] = (np.square(dx_p1 / dx_m1) - 1) * factor
        matrix[3, -3] = -np.square(dx_p1 / dx_m1) * factor
        # 1st order backward difference at upper boundary.
        dx = grid[-1] - grid[-2]
        matrix[2, -1] = 1 / dx
        matrix[3, -2] = -1 / dx
    else:
        raise ValueError(f"{band}: Unknown banded matrix. Use tri or penta.")
    return matrix


def d2dx2_equidistant(size: int,
                      dx: float,
                      band: str = "tri") -> np.ndarray:
    """FD approximation of 2nd order differential operator.

    Central finite difference approximation of 2nd order differential
    operator on equidistant grid. Linear boundary conditions are used.
    Assuming ascending grid.

    Args:
        size: Number of elements along main diagonal.
        dx: Constant grid spacing.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Discrete 2nd order differential operator.
    """
    if band == "tri":
        matrix = np.zeros((3, size))
        # 2nd order central difference.
        matrix[0, 2:] = 1
        matrix[1, 1:-1] = -2
        matrix[2, :-2] = 1
    elif band == "penta":
        matrix = np.zeros((5, size))
        # 2nd order central difference at next-to lower boundary.
        matrix[1, 2] = 1
        matrix[2, 1] = -2
        matrix[3, 0] = 1
        # 4th order central difference.
        matrix[0, 4:] = -1 / 12
        matrix[1, 3:-1] = 4 / 3
        matrix[2, 2:-2] = -5 / 2
        matrix[3, 1:-3] = 4 / 3
        matrix[4, :-4] = -1 / 12
        # 2nd order central difference at next-to upper boundary.
        matrix[1, -1] = 1
        matrix[2, -2] = -2
        matrix[3, -3] = 1
    else:
        raise ValueError(f"{band}: Unknown banded matrix. Use tri or penta.")
    return matrix / (dx ** 2)


def d2dx2(grid: np.ndarray,
          band: str = "tri") -> np.ndarray:
    """FD approximation of 2nd order differential operator.

    Finite difference approximation of 2nd order differential operator
    on non-equidistant grid. Linear boundary conditions are used.
    Assuming ascending grid.

    See Sundqvist & Veronis (1970).

    Args:
        grid: Grid points.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Discrete 2nd order differential operator.
    """
    if band == "tri":
        matrix = np.zeros((3, grid.size))
        # "2nd" order central difference.
        dx_p1 = grid[2:] - grid[1:-1]
        dx_m1 = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_p1 * dx_m1 * (1 + dx_p1 / dx_m1))
        matrix[0, 2:] = 2 * factor
        matrix[1, 1:-1] = -2 * (1 + dx_p1 / dx_m1) * factor
        matrix[2, :-2] = 2 * (dx_p1 / dx_m1) * factor
    elif band == "penta":
        matrix = np.zeros((5, grid.size))
        # "2nd" order central difference at next-to lower boundary.
        dx_p1 = grid[2] - grid[1]
        dx_m1 = grid[1] - grid[0]
        factor = 1 / (dx_p1 * dx_m1 * (1 + dx_p1 / dx_m1))
        matrix[1, 2] = 2 * factor
        matrix[2, 1] = -2 * (1 + dx_p1 / dx_m1) * factor
        matrix[3, 0] = 2 * (dx_p1 / dx_m1) * factor
        # "4th" order central difference.
        dx_p2 = grid[4:] - grid[3:-1]
        dx_p1 = grid[3:-1] - grid[2:-2]
        dx_m1 = grid[2:-2] - grid[1:-3]
        dx_m2 = grid[1:-3] - grid[:-4]
        factor = 1 / ((dx_m1 + dx_m2) * np.square(dx_p1 + dx_p2) / 2
                      - 16 * np.square(dx_p1) * dx_m1
                      - 16 * dx_p1 * np.square(dx_m1)
                      + np.square(dx_m1 + dx_m2) * (dx_p1 + dx_p2) / 2)
        matrix[0, 4:] = -(dx_m1 + dx_m2) * factor
        matrix[1, 3:-1] = 32 * dx_m1 * factor
        matrix[2, 2:-2] = -(-(dx_m1 + dx_m2)
                            + 32 * dx_m1 + 32 * dx_p1
                            - (dx_p1 + dx_p2)) * factor
        matrix[3, 1:-3] = 32 * dx_p1 * factor
        matrix[4, :-4] = -(dx_p1 + dx_p2) * factor
        # "2nd" order central difference at next-to upper boundary.
        dx_p1 = grid[-1] - grid[-2]
        dx_m1 = grid[-2] - grid[-3]
        factor = 1 / (dx_p1 * dx_m1 * (1 + dx_p1 / dx_m1))
        matrix[1, -3] = 2 * factor
        matrix[2, -2] = -2 * (1 + dx_p1 / dx_m1) * factor
        matrix[3, -1] = 2 * (dx_p1 / dx_m1) * factor
    else:
        raise ValueError(f"{band}: Unknown banded matrix. Use tri or penta.")
    return matrix


def d2dxdy(func: np.ndarray,
           ddx_x: np.ndarray,
           ddx_y: np.ndarray,
           band_x: str = "tri",
           band_y: str = "tri") -> np.ndarray:
    """FD approximation of 2nd order mixed differential operator.

    Args:
        func: Function values on 2-dimensional grid.
        ddx_x: FD approximation of 1st order differential operator in x
            dimension.
        ddx_y: FD approximation of 1st order differential operator in y
            dimension.
        band_x: Tri- or pentadiagonal matrix representation of
            operators in x dimension. Default is tridiagonal.
        band_y: Tri- or pentadiagonal matrix representation of
            operators in y dimension. Default is tridiagonal.

    Returns:
        Discrete 2nd order mixed differential operator.
    """
    matrix = np.zeros(func.shape)
    for idx_y in range(func.shape[1]):
        matrix[:, idx_y] = la.matrix_col_prod(ddx_x, func[:, idx_y], band_x)
    for idx_x in range(func.shape[0]):
        matrix[idx_x, :] = la.matrix_col_prod(ddx_y, matrix[idx_x, :], band_y)
    return matrix

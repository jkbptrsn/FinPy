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


def d2dxdy_equidistant(func: np.ndarray,
                       dx: float,
                       dy: float) -> np.ndarray:
    """FD approximation of 2nd order mixed differential operator.

    Central finite difference approximation of 2nd order mixed
    differential operator on equidistant grid. At the boundaries, 1st
    order forward/backward difference is used. Assuming ascending grid.

    Args:
        func: Function values on 2-dimensional grid.
        dx: Constant grid spacing in x-dimension.
        dy: Constant grid spacing in y-dimension.

    Returns:
        Discrete 2nd order mixed differential operator.
    """
    matrix = np.zeros(func.shape)
    # Interior points.
    for idx_y in range(1, func.shape[1] - 1):
        # (jxpjyp - jxmjyp - (jxpjym - jxmjym)) / (4 * dx * dy).
        matrix[1:-1, idx_y] = \
            (func[2:, idx_y + 1] - func[:-2, idx_y + 1]
             - (func[2:, idx_y - 1] - func[:-2, idx_y - 1])) / 4
    # Boundary points, jx = 0.
    # (jxpjyp - jxjyp - (jxpjym - jxjym)) / (2 * dx * dy).
    matrix[0, 1:-1] = \
        (func[1, 2:] - func[0, 2:] - (func[1, :-2] - func[0, :-2])) / 2
    # Boundary points, jx = -1.
    # (jxjyp - jxmjyp - (jxjym - jxmjym)) / (2 * dx * dy).
    matrix[-1, 1:-1] = \
        (func[-1, 2:] - func[-2, 2:] - (func[-1, :-2] - func[-2, :-2])) / 2
    # Boundary points, jy = 0.
    # (jxpjyp - jxpjy - (jxmjyp - jxmjy)) / (2 * dx * dy).
    matrix[1:-1, 0] = \
        (func[2:, 1] - func[2:, 0] - (func[:-2, 1] - func[:-2, 0])) / 2
    # Boundary points, jy = -1.
    # (jxpjy - jxpjym - (jxmjy - jxmjym)) / (2 * dx * dy).
    matrix[1:-1, -1] = \
        (func[2:, -1] - func[2:, -2] - (func[:-2, -1] - func[:-2, -2])) / 2
    # Corner points, jx = 0 and jy = 0.
    # (jxpjyp - jxjyp - (jxpjy - jxjy)) / (dx * dy).
    matrix[0, 0] = func[1, 1] - func[0, 1] - (func[1, 0] - func[0, 0])
    # Corner points, jx = 0 and jy = -1.
    # (jxpjy - jxjy - (jxpjym - jxjym)) / (dx * dy).
    matrix[0, -1] = func[1, -1] - func[0, -1] - (func[1, -2] - func[0, -2])
    # Corner points, jx = -1 and jy = 0.
    # (jxjyp - jxjy - (jxmjyp - jxmjy)) / (dx * dy).
    matrix[-1, 0] = func[-1, 1] - func[-1, 0] - (func[-2, 1] - func[-2, 0])
    # Corner points, jx = -1 and jy = -1.
    # (jxjy - jxjym - (jxmjy - jxmjym)) / (dx * dy).
    matrix[-1, -1] = \
        func[-1, -1] - func[-1, -2] - (func[-2, -1] - func[-2, -2])
    return matrix / (dx * dy)


def d2dxdy(func: np.ndarray,
           grid_x: np.ndarray,
           grid_y: np.ndarray) -> np.ndarray:
    """FD approximation of 2nd order mixed differential operator.

    Finite difference approximation of 2nd order mixed differential
    operator on non-equidistant grid. At the boundaries, 1st order
    forward/backward difference is used. Assuming ascending grid.

    Args:
        func: Function values on 2-dimensional grid.
        grid_x: Grid points in x dimension.
        grid_y: Grid points in x dimension.

    Returns:
        Discrete 2nd order mixed differential operator.
    """
    # Setting up finite difference factors.
    dx = grid_x[1:] - grid_x[0]
    dy = grid_y[1:] - grid_y[0]
    dxm = dx[:-1]
    dxp = dx[1:]
    dym = dy[:-1]
    dyp = dy[1:]
    ax = dxm / (dxp * (dxm + dxp))
    bx = - ax * (1 - (dxp / dxm) ** 2)
    cx = - ax * (dxp / dxm) ** 2
    ay = dym / (dyp * (dym + dyp))
    by = - ay * (1 - (dyp / dym) ** 2)
    cy = - ay * (dyp / dym) ** 2
    # Interior points.
    matrix = np.zeros(func.shape)
    for idx_y in range(1, func.shape[1] - 1):
        matrix[1:-1, idx_y] = \
            ax * ay[idx_y - 1] * func[2:, idx_y + 1] \
            + ax * by[idx_y - 1] * func[2:, idx_y] \
            + ax * cy[idx_y - 1] * func[2:, idx_y - 1] \
            + bx * ay[idx_y - 1] * func[1:-1, idx_y + 1] \
            + bx * by[idx_y - 1] * func[1:-1, idx_y] \
            + bx * cy[idx_y - 1] * func[1:-1, idx_y - 1] \
            + cx * ay[idx_y - 1] * func[:-2, idx_y + 1] \
            + cx * by[idx_y - 1] * func[:-2, idx_y] \
            + cx * cy[idx_y - 1] * func[:-2, idx_y - 1]
    # Boundary points, jx = 0.
    matrix[0, 1:-1] = \
        ay * (func[1, 2:] - func[0, 2:]) / dx[0] \
        + by * (func[1, 1:-1] - func[0, 1:-1]) / dx[0] \
        + cy * (func[1, :-2] - func[0, :-2]) / dx[0]
    # Boundary points, jx = -1.
    matrix[-1, 1:-1] = \
        ay * (func[-1, 2:] - func[-2, 2:]) / dx[-1] \
        + by * (func[-1, 1:-1] - func[-2, 1:-1]) / dx[-1] \
        + cy * (func[-1, :-2] - func[-2, :-2]) / dx[-1]
    # Boundary points, jy = 0.
    matrix[1:-1, 0] = \
        ax * (func[2:, 1] - func[2:, 0]) / dy[0] \
        + bx * (func[1:-1, 1] - func[1:-1, 0]) / dy[0] \
        + cx * (func[:-2, 1] - func[:-2, 0]) / dy[0]
    # Boundary points, jy = -1.
    matrix[1:-1, -1] = \
        ax * (func[2:, -1] - func[2:, -2]) / dy[-1] \
        + bx * (func[1:-1, -1] - func[1:-1, -2]) / dy[-1] \
        + cx * (func[:-2, -1] - func[:-2, -2]) / dy[-1]
    # Corner points, jx = 0 and jy = 0.
    # (jxpjyp - jxjyp - (jxpjy - jxjy)) / (dx * dy).
    matrix[0, 0] = func[1, 1] - func[0, 1] - (func[1, 0] - func[0, 0])
    matrix[0, 0] /= dx[0] * dy[0]
    # Corner points, jx = 0 and jy = -1.
    # (jxpjy - jxjy - (jxpjym - jxjym)) / (dx * dy).
    matrix[0, -1] = func[1, -1] - func[0, -1] - (func[1, -2] - func[0, -2])
    matrix[0, -1] /= dx[0] * dy[-1]
    # Corner points, jx = -1 and jy = 0.
    # (jxjyp - jxjy - (jxmjyp - jxmjy)) / (dx * dy).
    matrix[-1, 0] = func[-1, 1] - func[-1, 0] - (func[-2, 1] - func[-2, 0])
    matrix[-1, 0] /= dx[-1] * dy[0]
    # Corner points, jx = -1 and jy = -1.
    # (jxjy - jxjym - (jxmjy - jxmjym)) / (dx * dy).
    matrix[-1, -1] = \
        func[-1, -1] - func[-1, -2] - (func[-2, -1] - func[-2, -2])
    matrix[-1, -1] /= dx[-1] * dy[-1]
    return matrix


def d2dxdy_new(func: np.ndarray,
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
        matrix[idx_x, :] += la.matrix_col_prod(ddx_y, matrix[idx_x, :], band_y)
    return matrix

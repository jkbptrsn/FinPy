import numpy as np

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
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
    return matrix / dx


def ddx(grid: np.ndarray,
        band: str = "tri") -> np.ndarray:
    """FD approximation of 1st order differential operator.

    Finite difference approximation of 1st order differential operator
    on non-equidistant grid. At the boundaries, 1st order
    forward/backward difference is used. Assuming ascending grid.

    Same approximation is used for tri- and pentadiagonal form, see
    H. Sundqvist & G. Veronis, Tellus XXII (1970).

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
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * (1 + dx_plus / dx_minus))
        matrix[0, 2:] = factor
        matrix[1, 1:-1] = (np.square(dx_plus / dx_minus) - 1) * factor
        matrix[2, :-2] = - np.square(dx_plus / dx_minus) * factor
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
        # "2nd" order central difference.
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * (1 + dx_plus / dx_minus))
        matrix[1, 2:] = factor
        matrix[2, 1:-1] = (np.square(dx_plus / dx_minus) - 1) * factor
        matrix[3, :-2] = - np.square(dx_plus / dx_minus) * factor
        # 1st order backward difference at upper boundary.
        dx = grid[-1] - grid[-2]
        matrix[2, -1] = 1 / dx
        matrix[3, -2] = -1 / dx
    else:
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
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
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
    return matrix / (dx ** 2)


def d2dx2(grid: np.ndarray,
          band: str = "tri") -> np.ndarray:
    """FD approximation of 2nd order differential operator.

    Finite difference approximation of 2nd order differential operator
    on non-equidistant grid. Linear boundary conditions are used.
    Assuming ascending grid.

    Same approximation is used for tri- and pentadiagonal form, see
    H. Sundqvist & G. Veronis, Tellus XXII (1970).

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
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * dx_minus * (1 + dx_plus / dx_minus))
        matrix[0, 2:] = 2 * factor
        matrix[1, 1:-1] = - 2 * (1 + dx_plus / dx_minus) * factor
        matrix[2, :-2] = 2 * (dx_plus / dx_minus) * factor
    elif band == "penta":
        matrix = np.zeros((5, grid.size))
        # "2nd" order central difference.
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * dx_minus * (1 + dx_plus / dx_minus))
        matrix[1, 2:] = 2 * factor
        matrix[2, 1:-1] = - 2 * (1 + dx_plus / dx_minus) * factor
        matrix[3, :-2] = 2 * (dx_plus / dx_minus) * factor
    else:
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
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
           grid: np.ndarray) -> np.ndarray:
    """FD approximation of 2nd order mixed differential operator.

    Finite difference approximation of 2nd order mixed differential
    operator on non-equidistant grid. At the boundaries, 1st order
    forward/backward difference is used. Assuming ascending grid.

    Args:
        func: Function values on 2-dimensional grid.
        grid: Grid points.

    Returns:
        Discrete 2nd order mixed differential operator.
    """
    matrix = np.zeros(func.shape)
    # Interior points...
    return matrix

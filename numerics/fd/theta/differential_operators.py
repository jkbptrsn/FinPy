import numpy as np

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


def ddx_equidistant(n_elements: int,
                    dx: float,
                    form: str = "tri") -> np.ndarray:
    """Finite difference approximation of 1st order derivative operator.

    Central finite difference approximation of 1st order derivative
    operator on equidistant grid. At the boundaries, either 1st order
    (tri) or 2nd order (penta) forward/backward difference is used.
    TODO: Assuming ascending grid! Problem or just a comment...?

    Args:
        n_elements: Number of elements along main diagonal.
        dx: Equidistant spacing.
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Discrete 1st order derivative operator.
    """
    if form == "tri":
        matrix = np.zeros((3, n_elements))
        # Central difference.
        matrix[0, 2:] = 1
        matrix[2, :-2] = -1
        # Forward difference at lower boundary.
        matrix[0, 1] = 2
        matrix[1, 0] = -2
        # Backward difference at upper boundary.
        matrix[1, -1] = 2
        matrix[2, -2] = -2
    elif form == "penta":
        matrix = np.zeros((5, n_elements))
        # Central difference.
        matrix[1, 2:] = 1
        matrix[3, :-2] = -1
        # Forward difference at lower boundary.
        matrix[0, 2] = -1
        matrix[1, 1] = 4
        matrix[2, 0] = -3
        # Backward difference at upper boundary.
        matrix[2, -1] = 3
        matrix[3, -2] = -4
        matrix[4, -3] = 1
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix / (2 * dx)


def ddx(grid: np.ndarray,
        form: str = "tri") -> np.ndarray:
    """Finite difference approximation of 1st order derivative operator.

    Finite difference approximation of 1st order derivative operator on
    non-equidistant grid. At the boundaries, either 1st order (tri) or
    2nd order (penta) forward/backward difference is used.
    TODO: Assuming ascending grid! Problem or just a comment...?

    Args:
        grid: Grid in spatial dimension.
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Discrete 1st order derivative operator.
    """
    if form == "tri":
        matrix = np.zeros((3, grid.size))
        # "Central" difference.
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * (1 + dx_plus / dx_minus))
        matrix[0, 2:] = factor
        matrix[1, 1:-1] = (np.square(dx_plus / dx_minus) - 1) * factor
        matrix[2, :-2] = - np.square(dx_plus / dx_minus) * factor
        # Forward difference at lower boundary.
        dx = grid[1] - grid[0]
        matrix[0, 1] = 1 / dx
        matrix[1, 0] = -1 / dx
        # Backward difference at upper boundary.
        dx = grid[-1] - grid[-2]
        matrix[1, -1] = 1 / dx
        matrix[2, -2] = -1 / dx
    elif form == "penta":
        matrix = np.zeros((5, grid.size))
        # "Central" difference.
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * (1 + dx_plus / dx_minus))
        matrix[1, 2:] = factor
        matrix[2, 1:-1] = (np.square(dx_plus / dx_minus) - 1) * factor
        matrix[3, :-2] = - np.square(dx_plus / dx_minus) * factor
        # Forward difference at lower boundary.
        dx_p1 = grid[1] - grid[0]
        dx_p2 = grid[2] - grid[1]
        matrix[0, 2] = - dx_p1 / (dx_p2 * (dx_p1 + dx_p2))
        matrix[1, 1] = (dx_p1 + dx_p2) / (dx_p1 * dx_p2)
        matrix[2, 0] = (-2 * dx_p1 - dx_p2) / (dx_p2 * (dx_p1 + dx_p2))
        # Backward difference at upper boundary.
        dx_m1 = grid[-1] - grid[-2]
        dx_m2 = grid[-2] - grid[-3]
        matrix[2, -1] = (dx_m2 + 2 * dx_m1) / (dx_m1 * (dx_m1 + dx_m2))
        matrix[3, -2] = - (dx_m1 + dx_m2) / (dx_m1 * dx_m2)
        matrix[4, -3] = dx_m1 / (dx_m2 * (dx_m1 + dx_m2))
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix


def d2dx2_equidistant(n_elements: int,
                      dx: float,
                      form: str = "tri") -> np.ndarray:
    """Finite difference approximation of 2nd order derivative operator.

    Central finite difference approximation of 2nd order derivative
    operator on equidistant grid. At the boundaries, either linear
    boundary conditions (tri) or 1st order (penta) forward/backward
    difference is used.
    TODO: Assuming ascending grid! Problem or just a comment...?

    Args:
        n_elements: Number of elements along main diagonal.
        dx: Equidistant spacing.
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

    Returns:
        Discrete 2nd order derivative operator.
    """
    if form == "tri":
        matrix = np.zeros((3, n_elements))
        # Central difference.
        matrix[0, 2:] = 1
        matrix[1, 1:-1] = -2
        matrix[2, :-2] = 1
    elif form == "penta":
        matrix = np.zeros((5, n_elements))
        # Central difference.
        matrix[1, 2:] = 1
        matrix[2, 1:-1] = -2
        matrix[3, :-2] = 1
        # Forward difference at lower boundary.
        matrix[0, 2] = 1
        matrix[1, 1] = -2
        matrix[2, 0] = 1
        # Backward difference at upper boundary.
        matrix[2, -1] = 1
        matrix[3, -2] = -2
        matrix[4, -3] = 1
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix / (dx ** 2)


def d2dx2(grid: np.ndarray,
          form: str = "tri") -> np.ndarray:
    """Finite difference approximation of 2nd order derivative operator.

    Finite difference approximation of 2nd order derivative operator on
    noon-equidistant grid. At the boundaries, linear boundary conditions
    are used.
    TODO: Assuming ascending grid! Problem or just a comment...?

    Args:
        grid: Grid in spatial dimension.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Discrete 2nd order derivative operator.
    """
    if form == "tri":
        matrix = np.zeros((3, grid.size))
        # "Central" difference.
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * dx_minus * (1 + dx_plus / dx_minus))
        matrix[0, 2:] = 2 * factor
        matrix[1, 1:-1] = - 2 * (1 + dx_plus / dx_minus) * factor
        matrix[2, :-2] = 2 * (dx_plus / dx_minus) * factor
    elif form == "penta":
        matrix = np.zeros((5, grid.size))
        # "Central" difference.
        dx_plus = grid[2:] - grid[1:-1]
        dx_minus = grid[1:-1] - grid[:-2]
        factor = 1 / (dx_plus * dx_minus * (1 + dx_plus / dx_minus))
        matrix[1, 2:] = 2 * factor
        matrix[2, 1:-1] = - 2 * (1 + dx_plus / dx_minus) * factor
        matrix[3, :-2] = 2 * (dx_plus / dx_minus) * factor
        # Forward difference at lower boundary.
        # Backward difference at upper boundary.
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix


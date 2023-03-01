import numpy as np

info = """
    Use scipy to solve equation A x = b, where A is banded matrix.
        1. scipy.linalg.solve_banded
            Standard LU factorization of A.
        2. scipy.linalg.solveh_banded
            Uses Thomas' algorithm. Should only be used for Hermitian 
            positive-definite matrices (in this case, real symmetric 
            matrices with positive eigenvalues).

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


def identity_matrix(n_elements: int,
                    form: str = "tri") -> np.ndarray:
    """Identity matrix of banded form.

    Args:
        n_elements: Number of elements along diagonal.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Identity matrix.
    """
    if form == "tri":
        matrix = np.zeros((3, n_elements))
        matrix[1, :] = 1
    elif form == "penta":
        matrix = np.zeros((5, n_elements))
        matrix[2, :] = 1
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix


def matrix_col_prod(matrix: np.ndarray,
                    vector: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of banded matrix and column vector.

    Args:
        matrix: Banded matrix.
        vector: Column vector.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Matrix-column product as column vector.
    """
    if form == "tri":
        # Contribution from diagonal.
        product = matrix[1, :] * vector
        # Contribution from superdiagonal.
        product[:-1] += matrix[0, 1:] * vector[1:]
        # Contribution from subdiagonal.
        product[1:] += matrix[2, :-1] * vector[:-1]
    elif form == "penta":
        # Contribution from diagonal.
        product = matrix[2, :] * vector
        # Contribution from 2nd superdiagonal.
        product[:-2] += matrix[0, 2:] * vector[2:]
        # Contribution from 1st superdiagonal.
        product[:-1] += matrix[1, 1:] * vector[1:]
        # Contribution from 1st subdiagonal.
        product[1:] += matrix[3, :-1] * vector[:-1]
        # Contribution from 2nd subdiagonal.
        product[2:] += matrix[4, :-2] * vector[:-2]
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return product


def row_matrix_prod(vector: np.ndarray,
                    matrix: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of row vector and banded matrix.

    Args:
        vector: Row vector.
        matrix: Banded matrix.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Row-matrix product as row vector.
    """
    if form == "tri":
        # Contribution from diagonal.
        product = vector * matrix[1, :]
        # Contribution from superdiagonal.
        product[1:] += vector[:-1] * matrix[0, 1:]
        # Contribution from subdiagonal.
        product[:-1] += vector[1:] * matrix[2, :-1]
    elif form == "penta":
        # Contribution from diagonal.
        product = vector * matrix[2, :]
        # Contribution from 2nd superdiagonal.
        product[2:] += vector[:-2] * matrix[0, 2:]
        # Contribution from 1st superdiagonal.
        product[1:] += vector[:-1] * matrix[1, 1:]
        # Contribution from 1st subdiagonal.
        product[:-1] += vector[1:] * matrix[3, :-1]
        # Contribution from 2nd subdiagonal.
        product[:-2] += vector[2:] * matrix[4, :-2]
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return product


def dia_matrix_prod(diagonal: np.ndarray,
                    matrix: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of diagonal matrix and banded matrix.

    Args:
        diagonal: Diagonal matrix represented as vector.
        matrix: Banded matrix.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Banded matrix.
    """
    product = np.zeros(matrix.shape)
    if form == "tri":
        # Contribution from superdiagonal.
        product[0, 1:] = diagonal[:-1] * matrix[0, 1:]
        # Contribution from diagonal.
        product[1, :] = diagonal * matrix[1, :]
        # Contribution from subdiagonal.
        product[2, :-1] = diagonal[1:] * matrix[2, :-1]
    elif form == "penta":
        # Contribution from 2nd superdiagonal.
        product[0, 2:] = diagonal[:-2] * matrix[0, 2:]
        # Contribution from 1st superdiagonal.
        product[1, 1:] = diagonal[:-1] * matrix[1, 1:]
        # Contribution from diagonal.
        product[2, :] = diagonal * matrix[2, :]
        # Contribution from 1st subdiagonal.
        product[3, :-1] = diagonal[1:] * matrix[3, :-1]
        # Contribution from 2nd subdiagonal.
        product[4, :-2] = diagonal[2:] * matrix[4, :-2]
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return product


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
        form: Tri- or pentadiagonal form. Default is tridiagonal.

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
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Discrete 1st order derivative operator.
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


def delta_equidistant(vector: np.ndarray,
                      dx: float,
                      form: str = "tri") -> np.ndarray:
    """Finite difference calculation of delta.

    Assuming equidistant and ascending grid.

    Args:
        vector: ...
        dx: Equidistant spacing.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Delta
    """
    ddx = ddx_equidistant(vector.size, dx, form)
    return matrix_col_prod(ddx, vector, form)


def gamma_equidistant(vector: np.ndarray,
                      dx: float,
                      form: str = "tri") -> np.ndarray:
    """Finite difference calculation of gamma.

    Assuming equidistant and ascending grid.

    Args:
        vector: ...
        dx: Equidistant spacing.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Gamma
    """
    d2dx2 = d2dx2_equidistant(vector.size, dx, form)
    return matrix_col_prod(d2dx2, vector, form)

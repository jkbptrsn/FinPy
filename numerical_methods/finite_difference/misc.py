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
# TODO: Compare computation time for solve_banded and solveh_banded


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


# TODO: CONTINUE...
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
    pass


def dia_matrix_prod(diagonal: np.ndarray,
                    matrix: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of diagonal matrix and banded matrix.

    Args:
        diagonal: Diagonal matrix represented as vector.
        matrix: Banded matrix.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Product as banded matrix.
    """
    product = np.zeros(matrix.shape)
    if form == "tri":
        # Contribution from super-diagonal.
        product[0, 1:] = diagonal[:-1] * matrix[0, 1:]
        # Contribution from diagonal.
        product[1, :] = diagonal * matrix[1, :]
        # Contribution from sub-diagonal.
        product[2, :-1] = diagonal[1:] * matrix[2, :-1]
    elif form == "penta":
        # Contribution from 2nd super-diagonal.
        product[0, 2:] = diagonal[:-2] * matrix[0, 2:]
        # Contribution from 1st super-diagonal.
        product[1, 1:] = diagonal[:-1] * matrix[1, 1:]
        # Contribution from diagonal.
        product[2, :] = diagonal * matrix[2, :]
        # Contribution from 1st sub-diagonal.
        product[3, :-1] = diagonal[1:] * matrix[3, :-1]
        # Contribution from 2nd sub-diagonal.
        product[4, :-2] = diagonal[2:] * matrix[4, :-2]
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return product


def ddx_equidistant(n_elements: int,
                    dx: float,
                    form: str = "tri") -> np.ndarray:
    """Finite difference approximation of 1st order derivative operator.

    Central finite difference approximation of first order derivative
    operator. At the boundaries, first order forward/backward difference
    is used.

    Args:
        n_elements: Number of elements along main diagonal.
        dx: Equidistant spacing.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Discrete derivative operator.
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
        matrix[1, 1] = 2
        matrix[2, 0] = -2
        # Backward difference at upper boundary.
        matrix[2, -1] = 2
        matrix[3, -2] = -2
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix / (2 * dx)


def d2dx2_equidistant(n_elements: int,
                      dx: float,
                      form: str = "tri") -> np.ndarray:
    """Finite difference approximation of 2nd order derivative operator.

    Central finite difference approximation of 2nd order derivative
    operator. At the boundaries, either the operator is set equal to
    zero (tri) or 2nd order forward/backward difference is used (penta).

    Args:
        n_elements: Number of elements along main diagonal.
        dx: Equidistant spacing.
        form: Tri- or pentadiagonal form. Default is tridiagonal.

    Returns:
        Discrete derivative operator.
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

        # Backward difference at upper boundary.
    else:
        raise ValueError(
            f"{form}: Unknown form of banded matrix. Use tri or penta.")
    return matrix / (dx ** 2)


def delta(solution: np.ndarray,
          dx: float,
          form: str = "tri") -> np.ndarray:
    """Delta calculated by second order finite differences.

    Assuming equidistant and ascending grid.
    """
    delta = np.zeros(solution.shape)
    # Central finite difference.
    delta[1:-1] = (solution[2:] - solution[:-2]) / (2 * dx)
    # Forward finite difference.
    delta[0] = (- solution[2] / 2 + 2 * solution[1] - 3 * solution[0] / 2) / dx
    # Backward finite difference.
    delta[-1] = (solution[-3] / 2 - 2 * solution[-2] + 3 * solution[-1] / 2) / dx
    return delta

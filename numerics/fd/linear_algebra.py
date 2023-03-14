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


def identity_matrix(size: int,
                    band: str = "tri") -> np.ndarray:
    """Identity matrix of banded form.

    Args:
        size: Number of elements along main diagonal.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Identity matrix.
    """
    if band == "tri":
        matrix = np.zeros((3, size))
        matrix[1, :] = 1
    elif band == "penta":
        matrix = np.zeros((5, size))
        matrix[2, :] = 1
    else:
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
    return matrix


def matrix_col_prod(matrix: np.ndarray,
                    column: np.ndarray,
                    band: str = "tri") -> np.ndarray:
    """Product of banded matrix and column vector.

    Args:
        matrix: Banded matrix.
        column: Column vector.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Matrix-column product as column vector.
    """
    if band == "tri":
        # Contribution from main diagonal.
        product = matrix[1, :] * column
        # Contribution from superdiagonal.
        product[:-1] += matrix[0, 1:] * column[1:]
        # Contribution from subdiagonal.
        product[1:] += matrix[2, :-1] * column[:-1]
    elif band == "penta":
        # Contribution from diagonal.
        product = matrix[2, :] * column
        # Contribution from 2nd superdiagonal.
        product[:-2] += matrix[0, 2:] * column[2:]
        # Contribution from 1st superdiagonal.
        product[:-1] += matrix[1, 1:] * column[1:]
        # Contribution from 1st subdiagonal.
        product[1:] += matrix[3, :-1] * column[:-1]
        # Contribution from 2nd subdiagonal.
        product[2:] += matrix[4, :-2] * column[:-2]
    else:
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
    return product


def row_matrix_prod(row: np.ndarray,
                    matrix: np.ndarray,
                    band: str = "tri") -> np.ndarray:
    """Product of row vector and banded matrix.

    Args:
        row: Row vector.
        matrix: Banded matrix.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Row-matrix product as row vector.
    """
    if band == "tri":
        # Contribution from main diagonal.
        product = row * matrix[1, :]
        # Contribution from superdiagonal.
        product[1:] += row[:-1] * matrix[0, 1:]
        # Contribution from subdiagonal.
        product[:-1] += row[1:] * matrix[2, :-1]
    elif band == "penta":
        # Contribution from main diagonal.
        product = row * matrix[2, :]
        # Contribution from 2nd superdiagonal.
        product[2:] += row[:-2] * matrix[0, 2:]
        # Contribution from 1st superdiagonal.
        product[1:] += row[:-1] * matrix[1, 1:]
        # Contribution from 1st subdiagonal.
        product[:-1] += row[1:] * matrix[3, :-1]
        # Contribution from 2nd subdiagonal.
        product[:-2] += row[2:] * matrix[4, :-2]
    else:
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
    return product


def dia_matrix_prod(diagonal: np.ndarray,
                    matrix: np.ndarray,
                    band: str = "tri") -> np.ndarray:
    """Product of diagonal matrix and banded matrix.

    Args:
        diagonal: Diagonal matrix represented as vector.
        matrix: Banded matrix.
        band: Tri- or pentadiagonal matrix representation of operators.
            Default is tridiagonal.

    Returns:
        Banded matrix.
    """
    product = np.zeros(matrix.shape)
    if band == "tri":
        # Contribution from superdiagonal.
        product[0, 1:] = diagonal[:-1] * matrix[0, 1:]
        # Contribution from main diagonal.
        product[1, :] = diagonal * matrix[1, :]
        # Contribution from subdiagonal.
        product[2, :-1] = diagonal[1:] * matrix[2, :-1]
    elif band == "penta":
        # Contribution from 2nd superdiagonal.
        product[0, 2:] = diagonal[:-2] * matrix[0, 2:]
        # Contribution from 1st superdiagonal.
        product[1, 1:] = diagonal[:-1] * matrix[1, 1:]
        # Contribution from main diagonal.
        product[2, :] = diagonal * matrix[2, :]
        # Contribution from 1st subdiagonal.
        product[3, :-1] = diagonal[1:] * matrix[3, :-1]
        # Contribution from 2nd subdiagonal.
        product[4, :-2] = diagonal[2:] * matrix[4, :-2]
    else:
        raise ValueError(
            f"{band}: Unknown form of banded matrix. Use tri or penta.")
    return product

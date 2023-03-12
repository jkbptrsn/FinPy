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


def identity_matrix(n_elements: int,
                    form: str = "tri") -> np.ndarray:
    """Identity matrix of banded form.

    Args:
        n_elements: Number of elements along diagonal.
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

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
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

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
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

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
        form: Tri- ("tri") or pentadiagonal ("penta") form. Default
            is tridiagonal.

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

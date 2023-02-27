import numpy as np

info = """
    Tri-diagonal form:
        - 1st row: Super-diagonal (not including first element)
        - 2nd row: Diagonal
        - 3rd row: Sub-diagonal (not including last element)

    Penta-diagonal form:
        - 1st row: 2nd super-diagonal (not including first two elements)
        - 2nd row: 1st super-diagonal (not including first element)
        - 3rd row: Diagonal
        - 4th row: 1st sub-diagonal (not including last element)
        - 5th row: 2nd sub-diagonal (not including last two elements)
"""


def identity_matrix(n_elements: int,
                    form: str = "tri") -> np.ndarray:
    """Identity matrix of banded form.

    Args:
        n_elements: Number of elements along main diagonal.
        form: Tri- og penta-diagonal form. Default is tri-diagonal.

    Returns:
        Identity matrix.
    """
    if form == "tri":
        matrix = np.zeros((3, n_elements))
    elif form == "penta":
        matrix = np.zeros((5, n_elements))
    else:
        raise ValueError("Form of banded matrix is unknown: Use tri or penta.")
    matrix[1, :] = 1
    return matrix


def matrix_col_prod(matrix: np.ndarray,
                    vector: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of banded matrix and column vector.

    Args:
        matrix: Banded matrix.
        vector: Column vector.
        form: Tri- og penta-diagonal form. Default is tri-diagonal.

    Returns:
        Matrix-column product as column vector.
    """
    if form == "tri":
        # Contribution from diagonal.
        product = matrix[1, :] * vector
        # Contribution from super-diagonal.
        product[:-1] += matrix[0, 1:] * vector[1:]
        # Contribution from sub-diagonal.
        product[1:] += matrix[2, :-1] * vector[:-1]
    elif form == "penta":
        # Contribution from diagonal.
        product = matrix[2, :] * vector
        # Contribution from 2nd super-diagonal.
        product[:-2] += matrix[0, 2:] * vector[2:]
        # Contribution from 1st super-diagonal.
        product[:-1] += matrix[1, 1:] * vector[1:]
        # Contribution from 1st sub-diagonal.
        product[1:] += matrix[3, :-1] * vector[:-1]
        # Contribution from 2nd sub-diagonal.
        product[2:] += matrix[4, :-2] * vector[:-2]
    else:
        raise ValueError("Form of banded matrix is unknown: Use tri or penta.")
    return product


def row_matrix_prod(vector: np.ndarray,
                    matrix: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of row vector and banded matrix.

    Args:
        vector: Row vector.
        matrix: Banded matrix.
        form: Tri- og penta-diagonal form. Default is tri-diagonal.

    Returns:
        Row-matrix product as row vector.
    """
    pass


def dia_matrix_prod(diagonal: np.ndarray,
                    matrix: np.ndarray,
                    form: str = "tri") -> np.ndarray:
    """Product of diagonal matrix and banded matrix.

    Args:
        diagonal: Diagonal matrix.
        matrix: Banded matrix.
        form: Tri- og penta-diagonal form. Default is tri-diagonal.

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
        raise ValueError("Form of banded matrix is unknown: Use tri or penta.")
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
        form: Tri- og penta-diagonal form. Default is tri-diagonal.

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
        raise ValueError("Form of banded matrix is unknown: Use tri or penta.")
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
        form: Tri- og penta-diagonal form. Default is tri-diagonal.

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
        raise ValueError("Form of banded matrix is unknown: Use tri or penta.")
    return matrix / (dx ** 2)

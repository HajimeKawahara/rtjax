import numpy as np
import pytest


def generate_block_tridiagonal(matrix_diagonal, matrix_lower_diagonal,
                               matrix_upper_diagonal):
    """generates a block tridiagonal matrix given matrix_diagonal, matrix_lower_diagonal, matrix_upper_diagonal

    Args:
        matrix_diagonal (list of np.ndarray): List of square matrices for the main diagonal.
        matrix_lower_diagonal (list of np.ndarray): List of square matrices for the lower diagonal.
        matrix_upper_diagonal (list of np.ndarray): List of square matrices for the upper diagonal.

    Returns:
        np.matrix: The block tridiagonal matrix.
    """
    N = len(matrix_diagonal)

    zero_matrices = [np.zeros_like(matrix) for matrix in matrix_diagonal]
    block_matrix = []

    for i in range(N):
        row = []
        for j in range(N):
            if i == j:  # Diagonal
                row.append(matrix_diagonal[i])
            elif j == i - 1:  # Lower diagonal
                row.append(matrix_lower_diagonal[j])
            elif j == i + 1:  # Upper diagonal
                row.append(matrix_upper_diagonal[i])
            else:  # Everywhere else
                row.append(zero_matrices[i])
        block_matrix.append(row)

    block_tridiagonal_matrix = np.bmat(block_matrix)
    return block_tridiagonal_matrix


def decompose_block_diagonals_to_tridiagonals(diagonal_matrices):
    """
    Given N-block diagonal 2x2 matrix component of the block diagonal matrix, computes the main diagonal 'd', 
    the lower diagonal 'l', and the upper diagonal 'u' for the equivalent 2N x 2N tridiagonal matrix.

    Note:
        The block diagonal matrix L whose diagonal component consists of a 2x2 matrix can be regarded as a 2N x 2N tridiagonal matrix, 
        where N is the number of the diagonal components. This method decomposes L to the center, lower, and upper diagonals, d, l, and u. 

    Args:
        diagonal_matrices (ndarray): [N,2,2], block diagonal components (N block diagonal 2x2 matrix components of the block diagonal matrix)

    Returns:
        diagonal (ndarray): The main diagonal of the equivalent 2Nx2N tridiagonal matrix.
        lower_diagonal (ndarray): The lower diagonal of the equivalent 2Nx2N tridiagonal matrix.
        upper_diagonal (ndarray): The upper diagonal of the equivalent 2Nx2N tridiagonal matrix.

    Examples:

        >>> D0 = np.array([[1, 2], [3, 4]])
        >>> D1 = np.array([[5, 6], [7, 8]])
        >>> D2 = np.array([[9, 10], [11, 12]])
        >>> D3 = np.array([[13, 14], [15, 16]])
        >>> L = np.array([D0, D1, D2, D3])
        >>> d, l, u = decompose_block_diagonal_to_tridiagonal(L)
        >>> d #-> [1, 4, 5, 8, 9, 12, 13, 16]
        >>> l #-> [3, 0, 7, 0, 11, 0, 15]
        >>> u #-> [2, 0, 6, 0, 10, 0, 14]


    """
    nmat = len(diagonal_matrices)
    print(nmat)
    diagonal = np.diagonal(diagonal_matrices, axis1=1, axis2=2).flatten()
    lower_diagonal = np.insert(diagonal_matrices[:, 1, 0], range(1, nmat), 0)
    upper_diagonal = np.insert(diagonal_matrices[:, 0, 1], range(1, nmat), 0)

    return diagonal, lower_diagonal, upper_diagonal


import pytest
import numpy as np


def test_get_diagonals():
    D0 = np.array([[1, 2], [3, 4]])
    D1 = np.array([[5, 6], [7, 8]])
    D2 = np.array([[9, 10], [11, 12]])
    D3 = np.array([[13, 14], [15, 16]])
    L = np.array([D0, D1, D2, D3])

    d, l, u = decompose_block_diagonals_to_tridiagonals(L)
    print(d, l, u)
    np.testing.assert_array_equal(d, np.array([1, 4, 5, 8, 9, 12, 13, 16]))
    np.testing.assert_array_equal(l, np.array([3, 0, 7, 0, 11, 0, 15]))
    np.testing.assert_array_equal(u, np.array([2, 0, 6, 0, 10, 0, 14]))


def test_create_block_tridiagonal():
    D = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]])
    ]
    L = [np.array([[13, 14], [15, 16]]), np.array([[17, 18], [19, 20]])]
    U = [np.array([[21, 22], [23, 24]]), np.array([[25, 26], [27, 28]])]

    expected_result = np.bmat([[D[0], U[0], np.zeros_like(D[0])],
                               [L[0], D[1], U[1]],
                               [np.zeros_like(D[0]), L[1], D[2]]])

    np.testing.assert_array_equal(generate_block_tridiagonal(D, L, U),
                                  expected_result)


if __name__ == "__main__":
    #test_create_block_tridiagonal()
    test_get_diagonals()
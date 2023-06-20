import numpy as np
import pytest

def create_block_tridiagonal(matrix_diagonal, matrix_lower_diagonal, matrix_upper_diagonal):
    """Function to create a block tridiagonal matrix.

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

def decompose_block_diagonal(matrix_list):
    """
    Given a list of 2x2 matrices, computes the main diagonal 'd', 
    the lower diagonal 'l', and the upper diagonal 'u' for the equivalent 4x4 tridiagonal matrix.

    Parameters
    ----------
    matrix_list : ndarray
        The list of 2x2 matrices, N components, in nd 3D array form (N,2,2).

    Returns
    -------
    d : ndarray
        The main diagonal of the equivalent 2Nx2N tridiagonal matrix.
    l : ndarray
        The lower diagonal of the equivalent 2Nx2N tridiagonal matrix.
    u : ndarray
        The upper diagonal of the equivalent 2Nx2N tridiagonal matrix.

    """
    # The main diagonal is the diagonal of each 2x2 matrix
    d = np.diagonal(matrix_list, axis1=1, axis2=2).flatten()
    
    # The lower diagonal is the bottom-left of each 2x2 matrix,
    # with zeros in the positions corresponding to the zero blocks
    l = matrix_list[:, 1, 0]
    l = np.insert(l, range(1, len(l)), 0)
    # The upper diagonal is the top-right of each 2x2 matrix,
    # with zeros in the positions corresponding to the zero blocks
    u = matrix_list[:, 0, 1]
    u = np.insert(u, range(1, len(u)), 0)
    return d, l, u

import pytest
import numpy as np

def test_get_diagonals():
    D0 = np.array([[1, 2], [3, 4]])
    D1 = np.array([[5, 6], [7, 8]])
    D2 = np.array([[9, 10], [11, 12]])
    D3 = np.array([[13, 14], [15, 16]])
    L = [D0, D1, D2, D3]
    
    d, l, u = decompose_block_diagonal(L)
    print(d, l, u)
    np.testing.assert_array_equal(d, np.array([1, 4, 5, 8, 9, 12, 13, 16]))
    np.testing.assert_array_equal(l, np.array([3, 0, 7, 0, 11, 0, 15]))
    np.testing.assert_array_equal(u, np.array([2, 0, 6, 0, 10, 0, 14]))


def test_create_block_tridiagonal():
    D = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    L = [np.array([[13, 14], [15, 16]]), np.array([[17, 18], [19, 20]])]
    U = [np.array([[21, 22], [23, 24]]), np.array([[25, 26], [27, 28]])]

    expected_result = np.bmat([
        [D[0], U[0], np.zeros_like(D[0])],
        [L[0], D[1], U[1]],
        [np.zeros_like(D[0]), L[1], D[2]]
    ])

    np.testing.assert_array_equal(create_block_tridiagonal(D, L, U), expected_result)

if __name__ == "__main__":
    test_create_block_tridiagonal()
    test_get_diagonals()
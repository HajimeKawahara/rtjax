"""HBISA but not work...
"""
import numpy as np
import jax.numpy as jnp
from rtjax.btridiag import generate_block_tridiagonal


def _UplusLy_folded(L, U, y_folded):
    """compute (U+L)y 

    Args:
        L (_type_): _description_
        U (_type_): _description_
        y_folded (_type_): _description_

    Returns:
        _type_: _description_
    """
    O = np.zeros((1, 2))
    Ux = _parallel_mutmal(U, y_folded[1:, :])
    Ux = np.concatenate([Ux, O], axis=0)
    Lx = _parallel_mutmal(L, y_folded[:-1, :])
    Lx = np.concatenate([O, Lx], axis=0)
    return Ux + Lx


def _parallel_mutmal(A_folded, x_folded):
    """matmul for folded matrix and vector A@x

    Args:
        A_folded (_type_): (N, M, M) folded matrix
        x_folded (_type_): (N, M) folded vector

    Returns:
        folded matrix multiplied vector: (N, M)
    """
    return jnp.matmul(A_folded, x_folded[..., None])[..., 0]


def test_parallel_mutmul():
    D = np.array([
        np.array([[1., 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]])
    ])
    vector_folded = np.array([1., 2., 3., 4., 5., 6.]).reshape(3, 2)
    ans = _parallel_mutmal(D, vector_folded)
    for i in range(0, 3):
        assert np.sum(D[i, ...] @ vector_folded[i, ...] - ans[i, ...]) == 0.0


def test_UplusL():
    D = np.array([
        np.array([[1., 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]])
    ])
    L = np.array(
        [np.array([[13., 14], [15, 16]]),
         np.array([[17, 18], [19, 20]])])
    U = np.array(
        [np.array([[21., 22], [23, 24]]),
         np.array([[25, 26], [27, 28]])])
    N = len(D)
    Amat = generate_block_tridiagonal(D, L, U)
    y = np.array([1., 2., 3., 4., 5., 6.])
    val0 = Amat @ y
    y_folded = y.reshape(N, 2)
    UpLy = _UplusLy_folded(L, U, y_folded)
    val1 = _parallel_mutmal(D, y_folded) + UpLy
    assert np.sum((val0 - val1.flatten())**2) == 0.0


if __name__ == "__main__":
    test_parallel_mutmul()
    test_UplusL()
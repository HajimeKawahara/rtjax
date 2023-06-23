"""HBISA but very unstale
"""

import numpy as np
import jax.numpy as jnp
from rtjax.btridiag import generate_block_tridiagonal


def solve_block_tridiag_hbisa(D, L, U, vector_folded, BTD):
    """a block tridiagonal matrix solver using HBISA (Yang et al. J Supercomput 2017 73,1760-1781), A x = b.

    HBISA is a hybrid blocked iterative solving algorithm.
    
    Args:
        matrix_diagonal (list of 2x2 matrix): List of square matrices for the main diagonal .
        matrix_lower_diagonal (list of 2x2 matrix): List of square matrices for the lower diagonal.
        matrix_upper_diagonal (list of 2x2 matrix): List of square matrices for the upper diagonal.
        vector_folded: right-hand side vector b, but folded to (2, N)
        
    Returns:
        
    """
    Niter=100
    N = len(D)

    #iterative solver\
    x_folded = np.array([1.,2.,3.0001,4.,5.,6.]).reshape(N,2) #ans + delta -> unstable
    x_folded = np.array([1.,2.,3.,4.,5.,6.]).reshape(N,2) #ans -> stay answer
    Dinv = np.linalg.inv(D)
    
    #for debug
    Amat = generate_block_tridiagonal(D, L, U)
    
    for i in range(0,Niter):
        r_folded = vector_folded - _UplusLy_folded(L, U, x_folded)
        x_folded = _parallel_mutmal(Dinv, r_folded)
        
        #print(np.sum((Amat@x_folded.flatten() - vector_folded.flatten())**2))
        print("x=",x_folded.flatten())
        
    return x_folded

def _UplusLy_folded(L, U, y_folded):
    """compute (U+L)y 

    Args:
        L (_type_): _description_
        U (_type_): _description_
        y_folded (_type_): _description_

    Returns:
        _type_: _description_
    """
    O = np.zeros((1,2))
    Ux = _parallel_mutmal(U, y_folded[1:,:])
    Ux = np.concatenate([Ux, O], axis=0)
    Lx = _parallel_mutmal(L, y_folded[:-1,:])
    Lx = np.concatenate([O, Lx], axis=0)
    return Ux+Lx


def _parallel_mutmal(A_folded, x_folded):
    """matmul for folded matrix and vector A@x

    Args:
        A_folded (_type_): (N, M, M) folded matrix
        x_folded (_type_): (N, M) folded vector

    Returns:
        folded matrix multiplied vector: (N, M)
    """
    return jnp.matmul(A_folded,x_folded[...,None])[...,0]

def test_parallel_mutmul():
    D = np.array([np.array([[1., 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])])
    vector_folded = np.array([1.,2.,3.,4.,5.,6.]).reshape(3,2)
    ans = _parallel_mutmal(D, vector_folded)
    for i in range(0,3):
        assert np.sum(D[i,...]@vector_folded[i,...] - ans[i,...])==0.0

def test_UplusL():
    D = np.array([np.array([[1., 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])])
    L = np.array([np.array([[13., 14], [15, 16]]), np.array([[17, 18], [19, 20]])])
    U = np.array([np.array([[21., 22], [23, 24]]), np.array([[25, 26], [27, 28]])])
    N = len(D)
    Amat = generate_block_tridiagonal(D, L, U)
    y=np.array([1.,2.,3.,4.,5.,6.])
    val0=Amat@y
    y_folded=y.reshape(N,2)
    UpLy = _UplusLy_folded(L, U, y_folded)
    val1=_parallel_mutmal(D, y_folded) + UpLy
    assert np.sum((val0 - val1.flatten())**2) == 0.0


def test_solve_block_tridiag():
    from jax.config import config
    config.update("jax_enable_x64", True)
    D = np.array([np.array([[1., 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])])
    L = np.array([np.array([[13., 14], [15, 16]]), np.array([[17, 18], [19, 20]])])
    U = np.array([np.array([[21., 22], [23, 24]]), np.array([[25, 26], [27, 28]])])
    BTD = generate_block_tridiagonal(D, L, U)
    ans = np.array([1.,2.,3.,4.,5.,6.])
    print(BTD@ans)
    #vector_folded = np.array([1.,2.,3.,4.,5.,6.]).reshape(3,2)
    vector_folded = np.array([156., 176., 361., 403., 228., 264.]).reshape(3,2)
    x_folded = solve_block_tridiag_hbisa(D, L, U, vector_folded, BTD)
    x = x_folded.flatten()
    print(np.sum((BTD@x - vector_folded.flatten())**2))


if __name__ == "__main__":
    test_solve_block_tridiag()
    #test_parallel_mutmul()
    #test_UplusL()
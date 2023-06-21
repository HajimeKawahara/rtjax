""" Tridiagonal Solver 

- The original code of solve_tridiag was taken from lineax (https://github.com/google/lineax), under Apache 2.0 License (See LICENSES_bundled.txt).

"""
import numpy as np
import jax.numpy as jnp
from jax.lax import scan
from rtjax.btridiag import generate_block_tridiagonal
    
def solve_tridiag(diagonal, lower_diagonal, upper_diagonal, vector):
    """Tridiagonal Linear Solver for A x = b, using Thomas Algorithm  
    
    the original code was taken from lineax (https://github.com/google/lineax), under Apache 2.0 License (See LICENSES_bundled.txt).

    Args:
        diagonal (1D array): the diagonal component vector of a matrix A
        lower_diagonal (1D array): the lower diagonal component vector of a matrix A
        upper_diagonal (1D array): the upper diagonal component vector of a matrix A
        vector (1D array): the vector b

    Notes:
        notation from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        _p indicates prime, ie. `d_p` is the variable name for d' on wikipedia
        


    Returns:
        1D array: solution vector (x)
    """
        
        
    size = len(diagonal)

    def thomas_scan(prev_cd_carry, bd):
        c_p, d_p, step = prev_cd_carry
        # the index of `a` doesn't matter at step 0 as
        # we won't use it at all. Same for `c` at final step
        a_index = jnp.where(step > 0, step - 1, 0)
        c_index = jnp.where(step < size, step, 0)

        b, d = bd
        a, c = lower_diagonal[a_index], upper_diagonal[c_index]
        denom = b - a * c_p
        new_d_p = (d - a * d_p) / denom
        new_c_p = c / denom
        return (new_c_p, new_d_p, step + 1), (new_c_p, new_d_p)

    def backsub(prev_x_carry, cd_p):
        x_prev, step = prev_x_carry
        c_p, d_p = cd_p
        x_new = d_p - c_p * x_prev
        return (x_new, step + 1), x_new

    # not a dummy init! 0 is the proper value for all of these
    init_thomas = (0, 0, 0)
    init_backsub = (0, 0)
    diag_vec = (diagonal, vector)
    _, cd_p = scan(thomas_scan, init_thomas, diag_vec, unroll=32)
    _, solution = scan(backsub, init_backsub, cd_p, reverse=True, unroll=32)

    return solution

def solve_block_tridiag(D, L, U, vector_folded, BTD):
    """a block tridiagonal matrix solver using HBISA (Yang et al. J Supercomput 2017 73,1760-1781), A x = b.

    HBISA is a hybrid blocked iterative solving algorithm.
    
    Args:
        matrix_diagonal (list of 2x2 matrix): List of square matrices for the main diagonal .
        matrix_lower_diagonal (list of 2x2 matrix): List of square matrices for the lower diagonal.
        matrix_upper_diagonal (list of 2x2 matrix): List of square matrices for the upper diagonal.
        vector_folded: right-hand side vector b, but folded to (2, N)
        
    Returns:
        
    """
    from rtjax.btridiag import decompose_block_diagonals_to_tridiagonals
    Niter=10
    N = len(D)

    #settings
    diagonal, lower_diagonal, upper_diagonal = decompose_block_diagonals_to_tridiagonals(D)        
    #O = np.zeros((1,2,2))
    #Ltilde = np.concatenate([L, O], axis=0)

    #iterative solver\
    x_folded = np.zeros_like(vector_folded)
    O = np.zeros((1,2))
    
    #debug
    Ox = np.array([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
    Dmat = generate_block_tridiagonal(D, Ox, Ox)
    Amat = generate_block_tridiagonal(D, L, U)
    
    y=np.array([1.,2.,3.,4.,5.,6.])
    val0=Amat@y

    y_folded=y.reshape(N,2)
    Ux = _parallel_mutmal(U, y_folded[1:,:])
    Ux = np.concatenate([Ux, O], axis=0)
    Lx = _parallel_mutmal(L, y_folded[:-1,:])
    Lx = np.concatenate([O, Lx], axis=0)

    val1=_parallel_mutmal(D, y_folded) + Ux + Lx
    print(val0)
    print(val1)
    #import sys
    #sys.exit()
    
    for i in range(0,Niter):
        x_folded_prev = x_folded[:] #debug
        
        Ux = _parallel_mutmal(U, x_folded[1:, :])
        Ux = np.concatenate([Ux, O], axis=0)

        Lx = _parallel_mutmal(L, x_folded[:-1, :])
        Lx = np.concatenate([O, Lx], axis=0)
        
        r_folded = vector_folded - Lx - Ux 
        r = r_folded.flatten()

        x_folded = solve_tridiag(jnp.array(diagonal), jnp.array(lower_diagonal), jnp.array(upper_diagonal), jnp.array(r)).reshape(N,2)  

        #debug
        residuals_0 = Dmat@x_folded_prev.flatten()-r_folded.flatten()
        residuals_1 = Dmat@x_folded.flatten()-r_folded.flatten()
        #print(i, "residuals:",residuals_0,"->", residuals_1)
        #print(i, "x:",x_folded_prev.flatten(),"->", x_folded.flatten())
        print(Amat@x_folded.flatten() - vector_folded.flatten())
        #print("->",np.max(Dmat@x_folded.flatten()-r))
        
    return x_folded


def _parallel_mutmal(A_folded, x_folded):
    return jnp.matmul(A_folded,x_folded[...,None])[...,0]

def test_parallel_mutmul():
    D = np.array([np.array([[1., 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])])
    vector_folded = np.array([1.,2.,3.,4.,5.,6.]).reshape(3,2)
    ans = _parallel_mutmal(D, vector_folded)
    for i in range(0,3):
        assert np.sum(D[i,...]@vector_folded[i,...] - ans[i,...])==0.0


def test_solve_block_tridiag():
    D = np.array([np.array([[1., 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])])
    L = np.array([np.array([[13., 14], [15, 16]]), np.array([[17, 18], [19, 20]])])
    U = np.array([np.array([[21., 22], [23, 24]]), np.array([[25, 26], [27, 28]])])
    BTD = generate_block_tridiagonal(D, L, U)
    #vector_folded = np.array([1.,2.,3.,4.,5.,6.]).reshape(3,2)
    vector_folded = np.array([1.,0.,0.,0.,0.,0.]).reshape(3,2)
    x_folded = solve_block_tridiag(D, L, U, vector_folded, BTD)
    x = x_folded.flatten()
    print(BTD@x)

def test_solve_tridiag():
    diag = jnp.array([1.,2.,3.,4.])
    lower_diag = jnp.array([5.,6.,7.])
    upper_diag = jnp.array([8.,9.,10.])
    vector = jnp.array([4.,3.,2.,1.])

    x = solve_tridiag(diag, lower_diag, upper_diag, vector)

    mat = jnp.array([[1.,8.,0.,0.],
                     [5.,2.,9.,0.],
                     [0.,6.,3.,10.],
                     [0.,0.,7.,4.]])
    assert jnp.sum((mat@x - vector)**2) < 1.e-12

if __name__ == "__main__":
    #test_solve_tridiag()
    test_solve_block_tridiag()
    #test_parallel_mutmul()
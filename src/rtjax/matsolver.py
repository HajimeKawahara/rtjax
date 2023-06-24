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

def solve_block_tridiag(diagonals, lower_diagonals, upper_diagonals, vectors):
    """Block Tridiagonal Linear Solver for T x = d, using block Thomas Algorithm  
    
    the original tridiagonal system solver was taken from lineax (https://github.com/google/lineax), under Apache 2.0 License (See LICENSES_bundled.txt).

    Args:
        diagonals (3D array): the matrix diagonal component of a matrix T (B_0, B_1, ... ,B_N-1), [N, M, M]
        lower_diagonals (3D array): the lower matrix diagonal component vector of a matrix T (C_0, C_1, ... ,C_N-1) [N, M, M]
        upper_diagonals (3D array): the upper matrix diagonal component vector of a matrix T (A_0, A_1, ... ,A_N-1) [N, M, M]
        vectors (2D array): the vector of subvectors [N, M]

    Notes:
        shape(T) = [NM, NM] and the block matrix component has the shape of [M,M]
        notation from: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        _p indicates prime, ie. `d_p` is the variable name for d' on wikipedia
        


    Returns:
        2D array: solution (x) [N, M]
    """
    
    size = len(vectors)
    
    def thomas_scan(prev_cd_carry, bd):
        c_p, d_p, step = prev_cd_carry
        # the index of `a` doesn't matter at step 0 as
        # we won't use it at all. Same for `c` at final step
        a_index = jnp.where(step > 0, step - 1, 0)
        c_index = jnp.where(step < size, step, 0)

        b, d = bd
        a, c = lower_diagonals[a_index], upper_diagonals[c_index]
        gamma = b - a @ c_p
        #solve gamma new_d_p = d - a * d_p
        #new_d_p = (d - a * d_p) / gamma
        new_d_p = jnp.linalg.solve(gamma, d - a @ d_p)
        #solve gamma new_c_p = c 
        #new_c_p = c / gamma
        new_c_p = jnp.linalg.solve(gamma, c)

        return (new_c_p, new_d_p, step + 1), (new_c_p, new_d_p)

    def backsub(prev_x_carry, cd_p):
        x_prev, step = prev_x_carry
        c_p, d_p = cd_p
        x_new = d_p - c_p @ x_prev
        return (x_new, step + 1), x_new

    # not a dummy init! 0 is the proper value for all of these
    #init_thomas = (0, 0, 0)
    #init_backsub = (0, 0)
    zero_subvector = jnp.zeros_like(vectors[0])
    init_thomas = (zero_subvector, zero_subvector, 0)
    init_backsub = (zero_subvector, zero_subvector)
    
    
    diag_vec = (diagonals, vectors)
    _, cd_p = scan(thomas_scan, init_thomas, diag_vec, unroll=32)
    _, solution = scan(backsub, init_backsub, cd_p, reverse=True, unroll=32)

    return solution


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
    test_solve_tridiag()

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker


def make_tensor_decomposition(
    l_matrix, u_block_size=2,
    q_block_size=2,
    i_core=2,
    j_core=2,
    k_core=2
):
    """ 
    
    Make transformations and Tucker decomposition of L component  
  
    Parameters
    ----------
    l_matrix : ndarray
        L image component
    u_block_size : int
        size of U blocks
    q_block_size : int
        size of QxQ blocks
    i_core : int
        first dimension of core tensor
    j_core : int
        second dimension of core tensor
    k_core : int
        third dimension of core tensor
  
    Returns
    -------
    core : ndarray
    factors : ndarray list
  
    """
    l_matrix_shape = l_matrix.shape[0]
    u_matrix = (
        l_matrix
            .reshape((l_matrix_shape, l_matrix_shape))
            .reshape(
                l_matrix_shape // u_block_size,
                u_block_size,
                l_matrix_shape // u_block_size,
                u_block_size
            ).mean(axis=(1, 3))
    )
    n = (l_matrix_shape // u_block_size // q_block_size) ** 2
    block_matrix = (
        u_matrix
            .reshape(
                u_matrix.shape[0] // q_block_size,
                q_block_size, -1, q_block_size
            ).swapaxes(1, 2)
            .reshape(-1, q_block_size, q_block_size)
    )
    x = tl.tensor(block_matrix, dtype=tl.float32)
    return tucker(x, ranks=[i_core, j_core, k_core])


def make_hash(core, a_factor, b_factor, c_factor):
    a_p = a_factor.mean(axis=1)
    a_h = (a_p >= a_p.mean()).astype(int)
    b_p = b_factor.mean(axis=1)
    b_h = (b_p >= b_p.mean()).astype(int)
    c_p = c_factor.mean(axis=1)
    c_h = (c_p >= c_p.mean()).astype(int)   
    return np.hstack((a_h, b_h, c_h))


def tucker_hash(l_matrix, q_block_size):
    core, factors = make_tensor_decomposition(l_matrix, q_block_size=q_block_size)
    return make_hash(core, *factors)
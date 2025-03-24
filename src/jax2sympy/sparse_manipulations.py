"""
In this file I implement some utility functions for manipulating / combining
sparse matrices
"""
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp

def hstack(bcoo_list):
    """Horizontally concatenate BCOO matrices along columns (axis=1)."""
    return jsp.bcoo_concatenate(bcoo_list, dimension=1)

def vstack(bcoo_list):
    """Vertically concatenate BCOO matrices along rows (axis=0)."""
    return jsp.bcoo_concatenate(bcoo_list, dimension=0)

def bcoo_to_bcsr(bcoo: jsp.BCOO, block_shape=(1,1)):
    """
    Convert a BCOO matrix to a BCSR-like format with fixed block size (br, bc).
    
    Returns:
      blocks:    shape (num_blocks, br, bc) of the nonzero blocks
      col_blocks:1D int array (num_blocks,) of each block's column index
      indptr:    1D int array (R+1,) with row pointers for the blocks
      shape:     the overall (n, m) shape
      block_shape: (br, bc)
    """
    br, bc = block_shape
    n, m   = bcoo.shape
    R, C   = n // br, m // bc  # number of block-rows and block-columns
    
    # Identify which block each nonzero belongs to, and the offset within that block.
    i, j       = bcoo.indices[:, 0], bcoo.indices[:, 1]
    block_id   = (i // br) * C + (j // bc)   # flatten (row_block, col_block)
    i_mod, j_mod = i % br, j % bc
    
    # Group entries by block_id using jnp.unique(..., return_inverse=True).
    unique_ids, inv = jnp.unique(block_id, return_inverse=True)
    
    # Accumulate into block entries with scatter_add.
    blocks = jnp.zeros((unique_ids.size, br, bc), bcoo.data.dtype)
    blocks = blocks.at[inv, i_mod, j_mod].add(bcoo.data)
    
    # Extract each block's row and column index, then sort by row.
    row_blocks, col_blocks = unique_ids // C, unique_ids % C
    perm = jnp.argsort(row_blocks)
    row_blocks, col_blocks, blocks = row_blocks[perm], col_blocks[perm], blocks[perm]
    
    # Build the row pointer array (indptr) by counting how many blocks per row.
    row_counts = jnp.zeros(R, dtype=jnp.int32).at[row_blocks].add(1)
    indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(row_counts)])
    
    return blocks, col_blocks, indptr, (n, m), block_shape

if __name__ == "__main__":
    coords = jnp.array([[0,1], [1,0], [2,3], [5,5]])  # nonzero locations
    vals   = jnp.array([10.0, 20.0, 30.0, 40.0])
    mat    = jsp.BCOO((vals, coords), shape=(6,6))
    
    b = jnp.arange(6, dtype=jnp.float32)
    A = lambda x: mat @ x
    jax.scipy.sparse.linalg.bicgstab(A, b)

    pass
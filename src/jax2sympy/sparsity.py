"""
This file represents my motivation for building jax2sympy transpiler thing,
I wanted sparsity structure of arbitrary vector valued function jacobians and
hessians for large scale nonlinear programming in JAX. I demonstrate this by
creating the necessary sparse matrices for a 13D quadcopter MPC nonlinear optimization
problem. These

 I leave it here as an
example of what can be created with this tool.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sympy as sy
from jax2sympy.translate import jaxpr_to_sympy_expressions

def sym_sparse_jacobian(inp_shape, out_flat, out_shape, get_var_idx):
    """
    Function to perform the actual jacobian calculation inside of sympy using
    its diff function, this is inherently more efficient than doing the same in jax
    for a highly sparse jacobian as we dont calculate unecessary gradients, only
    the relevant ones in out_expr.free_symbols
    """
    get_var_idx = get_var_idx[0]
    sym_jac_val = []
    sym_jac_coo = []
    for out_idx, out_expr in enumerate(out_flat):
        x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')]
        out_expr_grad = [sy.diff(out_expr, x) for x in x_symbols]
        out_multi_idx = [int(i) for i in np.unravel_index(out_idx, out_shape)]
        for grad_value, symbol in zip(out_expr_grad, x_symbols):
            inp_flat_idx = get_var_idx[symbol]
            inp_multi_idx = [int(i) for i in np.unravel_index(inp_flat_idx, inp_shape)]
            sym_jac_val.append(grad_value)
            sym_jac_coo.append(out_multi_idx + inp_multi_idx)
    sym_jac_val = np.array(sym_jac_val)
    sym_jac_coo = np.array(sym_jac_coo)
    return sym_jac_val, sym_jac_coo

def sym_sparse_jaccoo(f, x):
    """
    Function to find the sparsity pattern of the jacobian of a jax function f
    with a given example input x. This is done symbolically and so the value of x
    does not matter, as long as it is the correct shape and dtype
    """

    jaxpr = jax.make_jaxpr(f)(x)

    out_syms, var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(jaxpr, var2sym=dict())
    out_syms = out_syms[0]

    num_inputs = len(jaxpr.jaxpr.invars)
    assert num_inputs == 1

    iterator = iter(var2sym)
    var = var2sym[next(iterator)]

    indices = [int(x.split('_')[-1]) for x in map(str, var)]
    get_var_idx = {k: v for k, v in zip(var, indices)}

    sym_out_flat = out_syms.flatten()

    jac_coo = []
    if len(sym_out_flat) == 1:
        out_expr = sym_out_flat[0]
        x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')] # when differentiated these terms dont go to zero
        jac_coo = jac_coo + [[get_var_idx[i]] for i in x_symbols]
    else:
        for out_idx, out_expr in enumerate(sym_out_flat):
            x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')] # when differentiated these terms dont go to zero
            jac_coo = jac_coo + [[out_idx, get_var_idx[i]] for i in x_symbols]
    jac_coo = np.array(jac_coo)

    return jac_coo

def jax_sparse_jacrev(f, jac_coo):
    """
    Returns a function `jacrev(x)` that computes the sparse Jacobian of `f`
    according to the given sparsity pattern `jac_coo`.
    
    Arguments:
      f: A function f(x) -> y (either scalar or vector-valued).
      jac_coo: Array of shape (N, 1) or (N, 2).
               If shape == (N, 1), we assume f is scalar-valued and each row is [col].
               If shape == (N, 2), each row is [row, col] for the partial d f[row] / d x[col].
    """
    jac_coo = jnp.array(jac_coo)
    ncols = jac_coo.shape[1]  # either 1 or 2

    def single_input_output_grad(x, coo):
        """
        Computes a single partial derivative:
          - If ncols == 1, interpret coo = [col], returning grad f(x) wrt x[col].
          - If ncols == 2, interpret coo = [row, col], returning grad f(x)[row] wrt x[col].
        """
        if ncols == 1:
            # Scalar function case: f: R^n -> R
            col = coo[0]
            def partial_func(x_col):
                x_reassembled = x.at[col].set(x_col)
                return f(x_reassembled)  # scalar output
        else:
            # Vector/matrix function case: f: R^n -> R^m
            row, col = coo
            def partial_func(x_col):
                x_reassembled = x.at[col].set(x_col)
                return f(x_reassembled)[row]  # pick out the row-th output

        # Take derivative w.r.t. x[col], evaluate at x[col]
        return jax.grad(partial_func)(x[col])

    def jacrev(x):
        # Vectorize over all coordinate pairs in jac_coo
        return jax.vmap(single_input_output_grad, in_axes=(None, 0))(x, jac_coo)

    return jacrev

def recursive_coo(coo1, coo2):
    """
    Function which takes in two coo sparse coordinate definitions from subsequent transforms
    and then combines them into one higher dimensional coo
    """
    return np.vstack([np.hstack([coo1[v], coo2[i,1:]]) for i, v in enumerate(coo2[:,0])])

def get_sparsity_pattern(f, x, type='hessian'): # or "jacobian"
    """
    A helper function doing setup and low level calls of other functions in the file
    to get the final desired sparsity pattern of a jacobian or hessian of a given function
    """
    if type == "jacobian":
        sym_jac_coo = sym_sparse_jaccoo(f, x)
        return sym_jac_coo

    if type == "hessian":
        jaxpr = jax.make_jaxpr(f)(x)
        out_syms, var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(jaxpr, var2sym=dict())

        num_inputs = len(jaxpr.jaxpr.invars)
        num_constants = len(jaxpr.jaxpr.constvars)
        num_outputs = len(jaxpr.jaxpr.outvars)

        iterator = iter(var2sym)
        # concatenate all variables and constants into data structures matching jax
        sym_inputs = [var2sym[next(iterator)] for i in range(num_inputs)]
        sym_constants = [var2sym[next(iterator)] for i in range(num_constants)]

        get_var_idx = [] # list of dictionaries for every variable
        in_shapes = []
        for var in sym_inputs:
            indices = [int(x.split('_')[-1]) for x in map(str, var)]
            get_var_idx.append({k: v for k, v in zip(var, indices)})
            in_shapes.append(list(var.shape))

        get_const_idx = []
        const_shapes = []
        for const in sym_constants:
            indices = [int(c.split('_')[-1]) for c in map(str, const)]
            get_const_idx.append({k: v for k, v in zip(const, indices)})
            const_shapes.append(list(const.shape))

        out_shapes = []
        for out_sym in out_syms:
            out_shapes.append(list(out_sym.shape))

        # sym_inp_flat = np.array(sym_inputs)[0].flatten()
        sym_inp_shape = np.array(sym_inputs)[0].shape
        sym_out_flat = np.array(out_syms[0]).flatten()
        sym_out_shape = np.array(out_syms[0]).shape
        
        sym_jac_val, sym_jac_coo = sym_sparse_jacobian(sym_inp_shape, sym_out_flat, sym_out_shape, get_var_idx)
        _, sym_hess_coo = sym_sparse_jacobian(sym_inp_shape, sym_jac_val, sym_jac_val.shape, get_var_idx)
        if sym_hess_coo.size == 0: # sometimes the hessian doesnt exist - think linear inequalities
            return None
        else:
            sym_hess_coo = recursive_coo(sym_jac_coo, sym_hess_coo) # map the coo's
            return sym_hess_coo

    else: raise Exception

def sparsify_nlp(f, h, g, x):
    """
    Function to retrieve all required sparse functions for SQP Interior Point Methods
    """

    # get sparsity patterns
    jac_f_coo = get_sparsity_pattern(f, x, type="jacobian")
    hes_f_coo = get_sparsity_pattern(f, x, type="hessian" )
    jac_h_coo = get_sparsity_pattern(h, x, type="jacobian")
    hes_h_coo = get_sparsity_pattern(h, x, type="hessian" )
    jac_g_coo = get_sparsity_pattern(g, x, type="jacobian")
    hes_g_coo = get_sparsity_pattern(g, x, type="hessian" )

    # transform the functions according to their patterns
    jac_f_sp = jax_sparse_jacrev(f, jac_f_coo)
    hes_f_sp = jax_sparse_jacrev(jac_f_sp, hes_f_coo)
    jac_h_sp = jax_sparse_jacrev(h, jac_h_coo)
    hes_h_sp = jax_sparse_jacrev(jac_h_sp, hes_h_coo)
    jac_g_sp = jax_sparse_jacrev(g, jac_g_coo)
    hes_g_sp = jax_sparse_jacrev(jac_g_sp, hes_g_coo)
    
    # we could jit those functions now - but left to a higher point in the code
    # we could also handle the conversion to the larger KKT system matrix coordinates here
    # but I think that is best served at a higher point in the code
    return jac_f_sp, jac_f_coo, jac_h_sp, jac_h_coo, jac_g_sp, jac_g_coo, \
        hes_f_sp, hes_f_coo, hes_h_sp, hes_h_coo, hes_g_sp, hes_g_coo,

if __name__ == "__main__":

    from problems import mpc

    f, h, g, x0, gt, aux = mpc.quadcopter_nav(N=3) # scales to at least N=500 - seems pretty linear, no strict tests

    jaxpr = jax.make_jaxpr(h)(x0)
    out_syms, var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(jaxpr)

    num_inputs = len(jaxpr.jaxpr.invars)
    num_constants = len(jaxpr.jaxpr.constvars)
    num_outputs = len(jaxpr.jaxpr.outvars)

    iterator = iter(var2sym)
    # concatenate all variables and constants into data structures matching jax
    sym_inputs = [var2sym[next(iterator)] for i in range(num_inputs)]
    sym_constants = [var2sym[next(iterator)] for i in range(num_constants)]

    get_var_idx = [] # list of dictionaries for every variable
    in_shapes = []
    for var in sym_inputs:
        indices = [int(x.split('_')[-1]) for x in map(str, var)]
        get_var_idx.append({k: v for k, v in zip(var, indices)})
        in_shapes.append(list(var.shape))

    get_const_idx = []
    const_shapes = []
    for const in sym_constants:
        indices = [int(c.split('_')[-1]) for c in map(str, const)]
        get_const_idx.append({k: v for k, v in zip(const, indices)})
        const_shapes.append(list(const.shape))

    out_shapes = []
    for out_sym in out_syms:
        out_shapes.append(list(out_sym.shape))

    # sym_inp_flat = np.array(sym_inputs)[0].flatten()
    sym_inp_shape = np.array(sym_inputs)[0].shape
    sym_out_flat = np.array(out_syms[0]).flatten()
    sym_out_shape = np.array(out_syms[0]).shape
    sym_jac_val, sym_jac_coo = sym_sparse_jacobian(sym_inp_shape, sym_out_flat, sym_out_shape, get_var_idx)
    sym_hess_val, sym_hess_coo = sym_sparse_jacobian(sym_inp_shape, sym_jac_val, sym_jac_val.shape, get_var_idx)
    sym_hess_coo = recursive_coo(sym_jac_coo, sym_hess_coo) # map the coo's
    
    sparsify_nlp(f, h, g, x0)

    jac_val, jac_coo = sym_sparse_jacrev(h, x0)
    hess_val, hess_coo = sym_sparse_jacrev(jac_val, x0)
    jacrev = jax_sparse_jacrev(h, jac_coo)
    pass


    jac_coo = sym_sparse_jaccoo(h, x0)
    jacrev = jax_sparse_jacrev(h, jac_coo)
    hess_coo = sym_sparse_jaccoo(jacrev, x0)

    # example usage for jacobian
    test = jax.jit(jacrev)(x0)


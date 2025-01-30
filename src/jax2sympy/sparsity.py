"""
This file represents my motivation for building jax2sympy transpiler thing,
I wanted sparsity structure of arbitrary vector valued function jacobians and
hessians for large scale nonlinear programming in JAX. I demonstrate this by
creating the necessary sparse matrices for a 13D quadcopter MPC nonlinear optimization
problem. The way these transforms is done is very inconsistent and not very performant,
but the resulting functions are the same performance as a better creation would be so
I am not so concerned.

I leave it here as an example of what can be created with this tool.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sympy as sy
from tqdm import tqdm
from jax2sympy.translate import jaxpr_to_sympy_expressions

#######################################
# ---------- SymPy sparsity --------- #
#######################################

def _sym_sparse_jacobian(inp_shape, out_flat, out_shape, get_var_idx):
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

def get_sparsity_pattern(f, x, type='hessian'): # "jacobian" or "hessian"
    """
    A helper function doing setup and low level calls of other functions in the file
    to get the final desired sparsity pattern of a jacobian or hessian of a given function
    """
    jaxpr = jax.make_jaxpr(f)(x)
    out_syms, var2sym, _, _ = jaxpr_to_sympy_expressions(jaxpr, var2sym=dict())

    if type == "jacobian":
        # calculate the sparsity pattern without actually calculating the jacobian
        out_syms = out_syms[0]
        num_inputs = len(jaxpr.jaxpr.invars)
        assert num_inputs == 1
        iterator = iter(var2sym)
        var = var2sym[next(iterator)]
        indices = [int(x.split('_')[-1]) for x in map(str, var)]
        get_var_idx = {k: v for k, v in zip(var, indices)}
        sym_out_flat = out_syms.flatten()
        sym_jac_coo = []
        if len(sym_out_flat) == 1:
            out_expr = sym_out_flat[0]
            x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')] # when differentiated these terms dont go to zero
            sym_jac_coo = sym_jac_coo + [[get_var_idx[i]] for i in x_symbols]
        else:
            for out_idx, out_expr in enumerate(sym_out_flat):
                x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')] # when differentiated these terms dont go to zero
                sym_jac_coo = sym_jac_coo + [[out_idx, get_var_idx[i]] for i in x_symbols]
        return np.array(sym_jac_coo)

    if type == "hessian":
        # calculate the sparsity pattern of the hessian by first creating the jacobian in sympy
        num_inputs = len(jaxpr.jaxpr.invars)
        num_constants = len(jaxpr.jaxpr.constvars)
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
        sym_inp_shape = np.array(sym_inputs)[0].shape
        sym_out_flat = np.array(out_syms[0]).flatten()
        sym_out_shape = np.array(out_syms[0]).shape
        sym_jac_val, sym_jac_coo = _sym_sparse_jacobian(sym_inp_shape, sym_out_flat, sym_out_shape, get_var_idx)
        _, sym_hess_coo = _sym_sparse_jacobian(sym_inp_shape, sym_jac_val, sym_jac_val.shape, get_var_idx)
        if sym_hess_coo.size == 0: # sometimes the hessian doesnt exist - think linear inequalities
            return None
        else:
            # map the coos recursively
            sym_hess_coo = np.vstack([np.hstack([sym_jac_coo[v], sym_hess_coo[i,1:]]) for i, v in enumerate(sym_hess_coo[:,0])])
            return sym_hess_coo

    else: raise Exception

#######################################
# --- JAX function transformation --- #
#######################################

def sparse_jacobian(f, jac_coo):
    """
    transforms dense f(x): R^n -> R^m into its sparse jacobian jac(x): R^n -> R^mxn given a known sparsity pattern of jac(x)
    """

    if jac_coo is None: # indicates that there isnt a function
        return lambda x: None # return a function that also returns None

    # adjust jac_coo to allow for jittability in partial when slicing y on return statement
    jac_coo = jnp.array(jac_coo, dtype=jnp.int32)
    nrows = jac_coo.shape[0]
    ncols = jac_coo.shape[1]
    jac_coo = jnp.hstack([jnp.zeros([nrows,1], dtype=jnp.int32), jac_coo]) if ncols == 1 else jac_coo
    
    def partial(element, coo):
        _x = x.at[coo[-1]].set(element)
        y = f(_x)
        if ncols == 1:
            return y
        elif ncols > 1:
            return y[coo[:-1]].squeeze()
    
    g = lambda element, coo: jax.grad(partial)(element[coo[-1]], coo)

    jacrev = lambda x: jax.vmap(g, in_axes=(None, 0))(x, jac_coo)
    
    return jacrev

def sparse_hessian(f, hes_coo):

    if hes_coo is None:
        # indicates that there isn't a function / or no known Hessian pattern
        return lambda x: None

    # Ensure hes_coo is a JAX array of int
    hes_coo = jnp.array(hes_coo, dtype=jnp.int32)
    ncols = hes_coo.shape[1]

    if ncols > 2:

        def partial_out_ij(x, out_idx, i, j):
            """
            Compute d^2 f_{out_idx}(x) / (dx_i dx_j).
            That is: first derivative w.r.t. x_i, then derivative of that result w.r.t. x_j.
            """

            # 1) Define a function g_i(u) = df_{out_idx}(u)/dx_i
            #    i.e., pick out the i-th component from jax.grad(f_{out_idx})(u).
            #    f_{out_idx}(u) means f(u)[out_idx].
            def g_i(u):
                return jax.grad(lambda z: f(z)[out_idx])(u)[i]

            # 2) Now take derivative of g_i w.r.t. x_j
            #    i.e. second partial derivative.
            return jax.grad(g_i)(x)[j]

        def hess_fn(x):
            """
            Evaluate all requested Hessian entries at x and return them as a 1D array.
            """
            def single_entry(coord):
                out_idx, i, j = coord
                return partial_out_ij(x, out_idx, i, j)

            # Vectorize over rows of hes_coo
            return jax.vmap(single_entry)(hes_coo)
        
    elif ncols == 2:

        def partial_ij(x, i, j):
            """
            Compute d^2 f(x) / (dx_i dx_j).
            That is: first derivative w.r.t. x_i, then derivative of that result w.r.t. x_j.
            """
            # 1) g_i(u) = derivative of f(u) w.r.t. x_i
            #    jax.grad(f)(u) is a vector, pick out component i
            def g_i(u):
                return jax.grad(f)(u)[i]

            # 2) derivative of g_i(u) w.r.t. x_j
            return jax.grad(g_i)(x)[j]

        def hess_fn(x):
            """
            Evaluate all requested Hessian entries at x and return them as a 1D array.
            """
            def single_entry(coord):
                i, j = coord
                return partial_ij(x, i, j)

            # Vectorize over rows of hes_coo
            return jax.vmap(single_entry)(hes_coo)

    return hess_fn

if __name__ == "__main__":

    from problems import mpc
    import matplotlib.pyplot as plt

    f, h, g, x, gt, aux = mpc.quadcopter_nav(N=3) # scales to at least N=500 - seems pretty linear, no strict tests

    jac_f_coo = get_sparsity_pattern(f, x, type="jacobian")
    hes_f_coo = get_sparsity_pattern(f, x, type="hessian" )
    jac_h_coo = get_sparsity_pattern(h, x, type="jacobian")
    hes_h_coo = get_sparsity_pattern(h, x, type="hessian" )
    jac_g_coo = get_sparsity_pattern(g, x, type="jacobian")
    hes_g_coo = get_sparsity_pattern(g, x, type="hessian" )

    jac_f_sp = sparse_jacobian(f, jac_f_coo)
    jac_h_sp = sparse_jacobian(h, jac_h_coo)
    jac_g_sp = sparse_jacobian(g, jac_g_coo)
    hes_f_sp = sparse_hessian(f, hes_f_coo)
    hes_h_sp = sparse_hessian(h, hes_h_coo)
    hes_g_sp = sparse_hessian(g, hes_g_coo)

    def get_dense(sp, coo, shape):
        jac_f_dense = np.zeros(shape)
        for _sp, _coo in zip(sp, coo):
            jac_f_dense[*_coo] = _sp
        return jac_f_dense
    
    def test_dense_jac(f, f_sp, coo, x):
        if coo is None: return None
        f_out = f(x)
        f_dense = get_dense(f_sp(x), coo, f_out.shape)
        discrepancy = np.max(np.abs(f_out - f_dense))
        assert discrepancy <= 1e-6

    def test_dense_hess(f, f_sp, coo, x, plot=False):
        if coo is None: return None
        f_out = f(x)
        f_dense = get_dense(f_sp(x), coo, f_out.shape)
        discrepancy = np.max(np.abs(f_out - f_dense))
        if plot is True:
            x_values = coo[:, 1]
            y_values = coo[:, 2]
            plt.figure()
            im1 = plt.imshow(jnp.sum(f_out, axis=0), cmap='viridis')  # You can change the colormap if needed
            plt.colorbar(im1)  # Add colorbar
            plt.savefig('test_ctrl.png', dpi=500)
            plt.close()
            plt.figure()
            im2 = plt.imshow(jnp.sum(f_dense, axis=0), cmap='viridis')
            plt.colorbar(im2)  # Add colorbar
            plt.scatter(x_values, y_values, marker='o', alpha=0.7)
            plt.savefig('test_hess_h_coo.png', dpi=500)
            plt.close()
        assert discrepancy <= 1e-6

    def test_coo(f, coos, x):
        if coos is None: return None
        outs = f(x)
        sum = np.array(0.)
        for coo in coos:
            sum += outs[*coo]
        assert np.max(np.abs(sum - outs.sum())) < 1e-5

    print("testing coos correctness...")
    test_coo(jax.jacrev(f), jac_f_coo, x)
    test_coo(jax.jacrev(h), jac_h_coo, x)
    test_coo(jax.jacrev(g), jac_g_coo, x)
    test_coo(jax.hessian(f), hes_f_coo, x)
    test_coo(jax.hessian(h), hes_h_coo, x)
    test_coo(jax.hessian(g), hes_g_coo, x)
    print('passed')

    print('testing jacobians correctness...')
    test_dense_jac(jax.jacrev(f), jac_f_sp, jac_f_coo, x)
    test_dense_jac(jax.jacrev(h), jac_h_sp, jac_h_coo, x)
    test_dense_jac(jax.jacrev(g), jac_g_sp, jac_g_coo, x)
    print('passed')

    print('testing hessian correctness...')
    test_dense_hess(jax.hessian(f), hes_f_sp, hes_f_coo, x)
    test_dense_hess(jax.hessian(h), hes_h_sp, hes_h_coo, x)
    test_dense_hess(jax.hessian(g), hes_g_sp, hes_g_coo, x)
    print('passed')

    # f = lambda x: jnp.array([x[0]*x[1], x[0]**2, jnp.sin(x[1]**x[0])])
    # jac_coo = get_sparsity_pattern(f, x, type="jacobian")
    # hes_coo = get_sparsity_pattern(f, x, type="hessian" )
    # x = jnp.array([1.,1.])
    # f_sp = sparse_jacobian(f, jac_coo)
    # # hes_f_sp = sparse_jacobian_sparse_matrix_input(f_sp, hes_coo)
    # hes_f_sp = sparse_hessian(f, hes_coo)
    # test_dense_hess(jax.hessian(f), hes_f_sp, hes_coo, x)

    # hes_h_sp = sparse_hessian(h, hes_h_coo)
    # # hes_h_sp = sparse_jacobian_sparse_matrix_input(jac_h_sp, hes_h_coo)# , x.shape, [*h(x).shape, *x.shape, *x.shape])
    # test_dense_hess(jax.hessian(h), hes_h_sp, hes_h_coo, x)


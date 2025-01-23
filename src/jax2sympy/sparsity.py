import numpy as np
import jax
import jax.numpy as jnp
import sympy as sy
from jax2sympy.translate import jaxpr_to_sympy_expressions

def sym_sparse_jacrev(f, x):
    """
    Function to find the symbolic jacobian and sparsity pattern of a jax function f
    with a given example input x. This is done symbolically and so the value of x
    does not matter, as long as it is the correct shape and dtype
    """

    jaxpr = jax.make_jaxpr(f)(x)

    out_syms, var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(jaxpr.jaxpr)
    out_syms = out_syms[0]

    num_inputs = len(jaxpr.jaxpr.invars)
    assert num_inputs == 1

    iterator = iter(var2sym)
    var = var2sym[next(iterator)]

    indices = [int(x.split('_')[-1]) for x in map(str, var)]
    get_var_idx = {k: v for k, v in zip(var, indices)}

    sym_inp_shape = var.shape
    sym_out_flat = out_syms.flatten()
    sym_out_shape = out_syms.shape

    jac_val = []
    jac_coo = []
    for out_idx, out_expr in enumerate(sym_out_flat):
        x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')]
        out_expr_grad = [sy.diff(out_expr, x) for x in x_symbols]
        out_multi_idx = [int(i) for i in np.unravel_index(out_idx, sym_out_shape)]
        for grad_value, symbol in zip(out_expr_grad, x_symbols):
            inp_flat_idx = get_var_idx[symbol]
            inp_multi_idx = [int(i) for i in np.unravel_index(inp_flat_idx, sym_inp_shape)]
            jac_val.append(grad_value)
            jac_coo.append(out_multi_idx + inp_multi_idx)
    jac_val = np.array(jac_val)
    jac_coo = np.array(jac_coo)

    return jac_val, jac_coo

def sym_sparse_jaccoo(f, x):
    """
    Function to find the sparsity pattern of the jacobian of a jax function f
    with a given example input x. This is done symbolically and so the value of x
    does not matter, as long as it is the correct shape and dtype
    """

    jaxpr = jax.make_jaxpr(f)(x)

    out_syms, var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(jaxpr)
    out_syms = out_syms[0]

    num_inputs = len(jaxpr.jaxpr.invars)
    assert num_inputs == 1

    iterator = iter(var2sym)
    var = var2sym[next(iterator)]

    indices = [int(x.split('_')[-1]) for x in map(str, var)]
    get_var_idx = {k: v for k, v in zip(var, indices)}

    sym_inp_shape = var.shape
    sym_out_flat = out_syms.flatten()
    sym_out_shape = out_syms.shape

    jac_coo = []
    for out_idx, out_expr in enumerate(sym_out_flat):
        x_symbols = [s for s in out_expr.free_symbols if str(s).startswith('x')] # when differentiated these terms dont go to zero
        jac_coo = jac_coo + [[out_idx, get_var_idx[i]] for i in x_symbols]
    jac_coo = np.array(jac_coo)

    return jac_coo

# now that we have the jac_coo - we want to create the jax differentiated function
def sparse_jacrev(f, jac_coo):

    jac_coo = jnp.array(jac_coo)

    def single_output_grad(x, coo):
        def partial_func(x_col):
            x_reassembled = x.at[coo[1]].set(x_col)
            return f(x_reassembled)[coo[0]]
        return jax.grad(partial_func)

    def jacrev(x):
        def single_input_output_grad(x, coo):
            return single_output_grad(x, coo)(x[coo[1]])
        return jax.vmap(single_input_output_grad, in_axes=(None, 0))(x, jac_coo)

    return jacrev

if __name__ == "__main__":

    from problems import mpc

    f, h, g, x0, gt, aux = mpc.quadcopter_nav(N=3) # N=100

    jac_coo = sym_sparse_jaccoo(h, x0)
    jacrev = sparse_jacrev(h, jac_coo)
    hess_coo = sym_sparse_jaccoo(jacrev, x0)

    # example usage for jacobian
    test = jax.jit(jacrev)(x0)


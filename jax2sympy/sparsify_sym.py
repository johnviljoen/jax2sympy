import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import sympy as sy
from jax2sympy.translate import jaxpr_to_sympy_expressions

#######################################
# ----- Symbolic Sparse Jacobian ---- #
#######################################

def _remove_unevaluated_derivatives(expr):
    """
    Replace unevaluated Derivative terms with 0.

    This handles cases like derivative of sign() which sympy can't evaluate.
    JAX treats these as 0 (or uses subgradients), so we do the same.
    """
    if not expr.has(sy.Derivative):
        return expr
    # Find all Derivative subexpressions and replace with 0
    derivs = list(expr.atoms(sy.Derivative))
    subs_dict = {d: 0 for d in derivs}
    return expr.subs(subs_dict)


def _sym_sparse_jacobian(inp_shape, out_flat, out_shape, get_var_idx, x_prefix='x0'):
    """
    Compute symbolic jacobian using sympy differentiation.

    Args:
        inp_shape: shape of the input to differentiate with respect to
        out_flat: flattened array of sympy expressions for outputs
        out_shape: shape of the output
        get_var_idx: dict mapping symbols to their flat indices
        x_prefix: prefix of symbols to differentiate with respect to (default 'x0')

    Returns:
        sym_jac_val: array of sympy expressions for non-zero jacobian entries
        sym_jac_coo: array of coordinates [out_idx, inp_idx] for each entry
    """
    get_var_idx = get_var_idx[0]
    sym_jac_val = []
    sym_jac_coo = []
    for out_idx, out_expr in enumerate(out_flat):
        # Filter for symbols belonging to the first input only (e.g., x0_0, x0_1, ...)
        x_symbols = [s for s in out_expr.free_symbols if str(s).startswith(x_prefix + '_')]
        out_expr_grad = [sy.diff(out_expr, x).doit() for x in x_symbols]
        out_multi_idx = [int(i) for i in np.unravel_index(out_idx, out_shape)]
        for grad_value, symbol in zip(out_expr_grad, x_symbols):
            # Handle unevaluated derivatives (e.g., derivative of sign)
            # by substituting 0 for the Derivative terms
            grad_value = _remove_unevaluated_derivatives(grad_value)
            # Skip entries that are exactly zero
            if grad_value == 0:
                continue
            inp_flat_idx = get_var_idx[symbol]
            inp_multi_idx = [int(i) for i in np.unravel_index(inp_flat_idx, inp_shape)]
            sym_jac_val.append(grad_value)
            sym_jac_coo.append(out_multi_idx + inp_multi_idx)
    sym_jac_val = np.array(sym_jac_val, dtype=object)
    sym_jac_coo = np.array(sym_jac_coo) if len(sym_jac_coo) > 0 else np.zeros((0, len(out_shape) + len(inp_shape)), dtype=np.int64)
    return sym_jac_val, sym_jac_coo


def _get_input_symbols(var2sym):
    """Extract first input's symbols and build index mapping from var2sym.

    Returns:
        var: the first input variable's symbol array
        get_var_idx: dict mapping symbols to their flat indices
        x_prefix: the prefix for this input's symbols (e.g., 'x0')
    """
    iterator = iter(var2sym)
    var = var2sym[next(iterator)]
    # Handle both 0-d and n-d arrays
    var_flat = np.atleast_1d(var).flatten()
    # Extract the prefix (e.g., 'x0' from 'x0_0')
    first_sym_str = str(var_flat[0])
    x_prefix = first_sym_str.rsplit('_', 1)[0]  # 'x0_0' -> 'x0'
    indices = [int(str(x).split('_')[-1]) for x in var_flat]
    get_var_idx = {k: v for k, v in zip(var_flat, indices)}
    return var, get_var_idx, x_prefix


def _extract_all_symbols(var2sym, jaxpr):
    """
    Extract all symbols (inputs and constants) from var2sym.

    Returns:
        all_input_symbols: list of flat arrays of sympy symbols for each input
        const_symbols: list of (symbol_name, const_value) pairs for all constants
    """
    num_inputs = len(jaxpr.jaxpr.invars)
    num_constants = len(jaxpr.jaxpr.constvars)

    iterator = iter(var2sym)
    input_vars = [var2sym[next(iterator)] for _ in range(num_inputs)]
    const_vars = [var2sym[next(iterator)] for _ in range(num_constants)]

    # Flatten input symbols for each input variable
    all_input_symbols = [np.atleast_1d(var).flatten() for var in input_vars]

    # Build constant symbol -> value mapping
    const_symbols = []
    for const_arr, const_val in zip(const_vars, jaxpr.consts):
        const_flat = np.atleast_1d(const_arr).flatten()
        const_val_flat = np.atleast_1d(const_val).flatten()
        for sym, val in zip(const_flat, const_val_flat):
            const_symbols.append((str(sym), float(val)))

    return all_input_symbols, const_symbols


def get_symbolic_jacobian(f, x, *args):
    """
    Get the symbolic jacobian expressions and sparsity pattern.

    Args:
        f: JAX function
        x: sample input array
        *args: additional arguments to f (for tracing)

    Returns:
        sym_jac_val: array of sympy expressions for non-zero entries
        sym_jac_coo: coordinate array of shape (nnz, 2) with [out_idx, inp_idx]
        all_input_symbols: list of flat arrays of input sympy symbols (x first, then args)
        const_symbols: list of (name, value) pairs for constant symbols
    """
    jaxpr = jax.make_jaxpr(f)(x, *args)
    out_syms, var2sym, _, _ = jaxpr_to_sympy_expressions(jaxpr, var2sym=dict())

    out_syms = out_syms[0]

    # Get first input (x) for jacobian differentiation
    var, get_var_idx, x_prefix = _get_input_symbols(var2sym)
    inp_shape = np.atleast_1d(var).shape
    sym_out_flat = np.atleast_1d(out_syms).flatten()
    sym_out_shape = np.atleast_1d(out_syms).shape

    sym_jac_val, sym_jac_coo = _sym_sparse_jacobian(
        inp_shape, sym_out_flat, sym_out_shape, [get_var_idx], x_prefix
    )

    # Extract all symbols including args and constants
    all_input_symbols, const_symbols = _extract_all_symbols(var2sym, jaxpr)

    return sym_jac_val, sym_jac_coo, all_input_symbols, const_symbols


def get_symbolic_hessian(f, x, *args):
    """
    Get the symbolic hessian expressions and sparsity pattern.

    Args:
        f: JAX function
        x: sample input array
        *args: additional arguments to f (for tracing)

    Returns:
        sym_hess_val: array of sympy expressions for non-zero entries
        sym_hess_coo: coordinate array of shape (nnz, 3) with [out_idx, i, j]
        all_input_symbols: list of flat arrays of input sympy symbols (x first, then args)
        const_symbols: list of (name, value) pairs for constant symbols
    """
    jaxpr = jax.make_jaxpr(f)(x, *args)
    out_syms, var2sym, _, _ = jaxpr_to_sympy_expressions(jaxpr, var2sym=dict())

    # Get first input (x) for hessian differentiation
    var, get_var_idx, x_prefix = _get_input_symbols(var2sym)
    inp_shape = np.atleast_1d(var).shape
    sym_out_flat = np.atleast_1d(out_syms[0]).flatten()
    sym_out_shape = np.atleast_1d(out_syms[0]).shape

    # Extract all symbols including args and constants
    all_input_symbols, const_symbols = _extract_all_symbols(var2sym, jaxpr)

    # First differentiation: get jacobian
    sym_jac_val, sym_jac_coo = _sym_sparse_jacobian(
        inp_shape, sym_out_flat, sym_out_shape, [get_var_idx], x_prefix
    )

    if len(sym_jac_val) == 0:
        return np.array([], dtype=object), np.zeros((0, 3), dtype=np.int64), all_input_symbols, const_symbols

    # Second differentiation: get hessian
    sym_hess_val, sym_hess_coo = _sym_sparse_jacobian(
        inp_shape, sym_jac_val, sym_jac_val.shape, [get_var_idx], x_prefix
    )

    if sym_hess_coo.size == 0:
        return np.array([], dtype=object), np.zeros((0, 3), dtype=np.int64), all_input_symbols, const_symbols

    # Map coordinates: hess_coo[:,0] indexes into jac_coo, need to expand
    # sym_hess_coo is [jac_entry_idx, inp_idx]
    # We need [out_idx, inp_idx_i, inp_idx_j] where:
    #   - out_idx comes from sym_jac_coo[jac_entry_idx, 0]
    #   - inp_idx_i comes from sym_jac_coo[jac_entry_idx, 1]
    #   - inp_idx_j comes from sym_hess_coo[:, 1]
    full_hess_coo = np.vstack([
        np.hstack([sym_jac_coo[v], sym_hess_coo[i, 1:]])
        for i, v in enumerate(sym_hess_coo[:, 0])
    ])

    return sym_hess_val, full_hess_coo, all_input_symbols, const_symbols


#######################################
# ----- JAX Function Generation ----- #
#######################################

def sparse_jacobian_sym(f, x, *sample_args, coo_pattern=None, out_shape=None):
    """
    Create a sparse jacobian function using symbolic differentiation.

    Instead of using JAX autodiff (vmap(grad)), this computes derivatives
    symbolically in sympy and converts them to JAX functions.

    Args:
        f: JAX function
        x: sample input array
        *sample_args: sample additional arguments (for tracing)
        coo_pattern: optional COO pattern array of shape (nnz, 2) with [out_idx, x_idx].
                    If None, computed from symbolic differentiation.
                    If provided, symbolic expressions are filtered to match this pattern.
        out_shape: optional output shape (m, n). If None, inferred from function and x.

    Returns:
        jac_fn: function that takes (x, *args) and returns BCOO sparse jacobian
    """
    from sympy2jax import SymbolicModule

    sym_jac_val, sym_jac_coo, all_input_symbols, const_symbols = get_symbolic_jacobian(f, x, *sample_args)

    # Determine output shape
    if out_shape is None:
        out_shape = (f(x, *sample_args).size, x.size)

    # Handle provided COO pattern
    if coo_pattern is not None:
        coo_pattern = np.atleast_2d(coo_pattern)
        if len(coo_pattern) == 0 or coo_pattern.size == 0:
            return lambda x, *args: BCOO(
                (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)),
                shape=out_shape
            )

        # Handle legacy 1-col COO for scalar functions: [x_idx] without out_idx
        # The original sparse_jacobian handles this by stripping the leading 1 from out_shape
        is_legacy_1col = coo_pattern.shape[1] == 1
        if is_legacy_1col and out_shape[0] == 1:
            out_shape = out_shape[1:]

        # Create mapping from symbolic COO to expression index
        coo_to_idx = {tuple(coo): i for i, coo in enumerate(sym_jac_coo)}

        # Build filtered expressions for provided pattern
        filtered_val = []
        for coo in coo_pattern:
            # For legacy 1-col, prepend 0 to match 2-col symbolic format
            lookup_coo = tuple([0] + list(coo)) if is_legacy_1col else tuple(coo)
            idx = coo_to_idx.get(lookup_coo)
            if idx is not None:
                filtered_val.append(sym_jac_val[idx])
            else:
                # Entry not in symbolic pattern - this is zero
                filtered_val.append(sy.Integer(0))
        sym_jac_val = np.array(filtered_val, dtype=object)
        sym_jac_coo = coo_pattern

    if len(sym_jac_coo) == 0:
        return lambda x, *args: BCOO(
            (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)),
            shape=out_shape
        )

    jac_coo = jnp.array(sym_jac_coo, dtype=jnp.int32)

    # Build symbol name lists for each input
    all_sym_names = [[str(s) for s in syms] for syms in all_input_symbols]

    # Convert sympy expressions to JAX function via sympy2jax
    sym_module = SymbolicModule(list(sym_jac_val))

    # Pre-build constant dict (these don't change)
    const_dict = {name: val for name, val in const_symbols}

    def jac_fn(x, *args):
        # Create symbol -> value mapping for all inputs
        sym_dict = {}
        # First input is x
        x_flat = jnp.atleast_1d(x).flatten()
        for i, name in enumerate(all_sym_names[0]):
            sym_dict[name] = x_flat[i]
        # Additional args - only process args that have corresponding symbols
        # (runtime may pass more args than were traced, but unused args won't have symbols)
        num_traced_args = len(all_sym_names) - 1  # subtract 1 for x
        for arg_idx, arg in enumerate(args[:num_traced_args]):
            arg_flat = jnp.atleast_1d(arg).flatten()
            for i, name in enumerate(all_sym_names[arg_idx + 1]):
                sym_dict[name] = arg_flat[i]
        # Add constants
        sym_dict.update(const_dict)
        # Evaluate all jacobian entries
        jac_vals = sym_module(**sym_dict)
        return BCOO((jnp.array(jac_vals), jac_coo), shape=out_shape)

    return jac_fn


def sparse_hessian_sym(f, x, *sample_args, coo_pattern=None, out_shape=None):
    """
    Create a sparse hessian function using symbolic differentiation.

    Instead of using JAX autodiff (vmap(grad(grad))), this computes second
    derivatives symbolically in sympy and converts them to JAX functions.

    Args:
        f: JAX function
        x: sample input array
        *sample_args: sample additional arguments (for tracing)
        coo_pattern: optional COO pattern array of shape (nnz, 3) with [out_idx, i, j].
                    If None, computed from symbolic differentiation.
                    If provided, symbolic expressions are filtered to match this pattern.
        out_shape: optional output shape (m, n, n). If None, inferred from function and x.

    Returns:
        hess_fn: function that takes (x, *args) and returns BCOO sparse hessian
    """
    from sympy2jax import SymbolicModule

    sym_hess_val, sym_hess_coo, all_input_symbols, const_symbols = get_symbolic_hessian(f, x, *sample_args)

    # Determine output shape
    if out_shape is None:
        out_shape = (f(x, *sample_args).size, x.size, x.size)

    # Handle provided COO pattern
    if coo_pattern is not None:
        coo_pattern = np.atleast_2d(coo_pattern)
        if len(coo_pattern) == 0 or coo_pattern.size == 0:
            return lambda x, *args: BCOO(
                (jnp.array([]), jnp.zeros((0, 3), dtype=jnp.int32)),
                shape=out_shape
            )

        # Handle legacy 2-col COO for scalar functions: [i, j] without out_idx
        # The original sparse_hessian handles this by stripping the leading 1 from out_shape
        is_legacy_2col = coo_pattern.shape[1] == 2
        if is_legacy_2col and out_shape[0] == 1:
            out_shape = out_shape[1:]

        # Create mapping from symbolic COO to expression index
        # For 2-col provided COO, we need to match against 3-col symbolic COO by prepending 0
        coo_to_idx = {tuple(coo): i for i, coo in enumerate(sym_hess_coo)}

        # Build filtered expressions for provided pattern
        filtered_val = []
        for coo in coo_pattern:
            # For legacy 2-col, prepend 0 to match 3-col symbolic format
            lookup_coo = tuple([0] + list(coo)) if is_legacy_2col else tuple(coo)
            idx = coo_to_idx.get(lookup_coo)
            if idx is not None:
                filtered_val.append(sym_hess_val[idx])
            else:
                # Entry not in symbolic pattern - this is zero
                filtered_val.append(sy.Integer(0))
        sym_hess_val = np.array(filtered_val, dtype=object)
        sym_hess_coo = coo_pattern

    if len(sym_hess_coo) == 0:
        return lambda x, *args: BCOO(
            (jnp.array([]), jnp.zeros((0, 3), dtype=jnp.int32)),
            shape=out_shape
        )

    hess_coo = jnp.array(sym_hess_coo, dtype=jnp.int32)

    # Build symbol name lists for each input
    all_sym_names = [[str(s) for s in syms] for syms in all_input_symbols]

    # Convert sympy expressions to JAX function
    sym_module = SymbolicModule(list(sym_hess_val))

    # Pre-build constant dict (these don't change)
    const_dict = {name: val for name, val in const_symbols}

    def hess_fn(x, *args):
        # Create symbol -> value mapping for all inputs
        sym_dict = {}
        # First input is x
        x_flat = jnp.atleast_1d(x).flatten()
        for i, name in enumerate(all_sym_names[0]):
            sym_dict[name] = x_flat[i]
        # Additional args - only process args that have corresponding symbols
        # (runtime may pass more args than were traced, but unused args won't have symbols)
        num_traced_args = len(all_sym_names) - 1  # subtract 1 for x
        for arg_idx, arg in enumerate(args[:num_traced_args]):
            arg_flat = jnp.atleast_1d(arg).flatten()
            for i, name in enumerate(all_sym_names[arg_idx + 1]):
                sym_dict[name] = arg_flat[i]
        # Add constants
        sym_dict.update(const_dict)
        # Evaluate all hessian entries
        hess_vals = sym_module(**sym_dict)
        return BCOO((jnp.array(hess_vals), hess_coo), shape=out_shape)

    return hess_fn


#######################################
# ------------ Testing -------------- #
#######################################

if __name__ == "__main__":
    from jax2sympy.sparsify import test_dense_jac, test_dense_hess, get_dense
    from datetime import datetime

    # Simple test functions first
    print("=== Simple Function Tests ===")

    def f_simple(x):
        return jnp.array([x[0] * x[1], x[0]**2, jnp.sin(x[1])])

    x_simple = jnp.array([1.0, 2.0])

    print("Testing simple jacobian...")
    sym_jac_val, sym_jac_coo, input_syms, const_syms = get_symbolic_jacobian(f_simple, x_simple)
    print(f"  Jacobian entries: {len(sym_jac_val)}")
    print(f"  Jacobian COO shape: {sym_jac_coo.shape}")
    print(f"  Sample expressions: {sym_jac_val[:3] if len(sym_jac_val) >= 3 else sym_jac_val}")

    jac_simple_fn = sparse_jacobian_sym(f_simple, x_simple)
    result = jac_simple_fn(x_simple)
    jac_dense = jax.jacrev(f_simple)(x_simple)
    print(f"  Sparse result data: {result.data}")
    print(f"  Dense jac:\n{jac_dense}")
    test_dense_jac(jax.jacrev(f_simple), jac_simple_fn, sym_jac_coo, x_simple)
    print("  Simple jacobian passed!")

    print("\nTesting simple hessian...")
    def f_scalar(x):
        return x[0]**2 * x[1] + jnp.sin(x[0] * x[1])

    sym_hess_val, sym_hess_coo, _, _ = get_symbolic_hessian(f_scalar, x_simple)
    print(f"  Hessian entries: {len(sym_hess_val)}")
    print(f"  Hessian COO shape: {sym_hess_coo.shape}")

    hess_simple_fn = sparse_hessian_sym(f_scalar, x_simple)
    hess_result = hess_simple_fn(x_simple)
    hess_dense = jax.hessian(f_scalar)(x_simple)
    print(f"  Sparse hess data: {hess_result.data}")
    print(f"  Dense hess:\n{hess_dense}")
    test_dense_hess(jax.hessian(f_scalar), hess_simple_fn, sym_hess_coo, x_simple)
    print("  Simple hessian passed!")

    print("\n=== MPC Problem Tests ===")
    from jax2sympy.problems import mpc
    print("Loading problem...")
    f, h, g, x, gt, aux = mpc.quadcopter_nav(N=3)

    print("\n=== Testing Symbolic Jacobians ===")

    t1 = datetime.now()
    jac_f_sym = sparse_jacobian_sym(f, x)
    jac_h_sym = sparse_jacobian_sym(h, x)
    jac_g_sym = sparse_jacobian_sym(g, x)
    t2 = datetime.now()
    print(f"Time to create symbolic sparse jacobians: {t2-t1}")

    # Get the COO patterns for testing
    _, jac_f_coo, _, _ = get_symbolic_jacobian(f, x)
    _, jac_h_coo, _, _ = get_symbolic_jacobian(h, x)
    _, jac_g_coo, _, _ = get_symbolic_jacobian(g, x)

    print("Testing jacobian correctness...")
    test_dense_jac(jax.jacrev(f), jac_f_sym, jac_f_coo, x)
    test_dense_jac(jax.jacrev(h), jac_h_sym, jac_h_coo, x)
    test_dense_jac(jax.jacrev(g), jac_g_sym, jac_g_coo, x)
    print("Jacobians passed!")

    print("\n=== Testing Symbolic Hessians ===")

    t1 = datetime.now()
    hes_f_sym = sparse_hessian_sym(f, x)
    hes_h_sym = sparse_hessian_sym(h, x)
    hes_g_sym = sparse_hessian_sym(g, x)
    t2 = datetime.now()
    print(f"Time to create symbolic sparse hessians: {t2-t1}")

    # Get the COO patterns for testing
    _, hes_f_coo, _, _ = get_symbolic_hessian(f, x)
    _, hes_h_coo, _, _ = get_symbolic_hessian(h, x)
    _, hes_g_coo, _, _ = get_symbolic_hessian(g, x)

    print("Testing hessian correctness...")
    test_dense_hess(jax.hessian(f), hes_f_sym, hes_f_coo, x)
    test_dense_hess(jax.hessian(h), hes_h_sym, hes_h_coo, x)
    test_dense_hess(jax.hessian(g), hes_g_sym, hes_g_coo, x)
    print("Hessians passed!")

    print("\nAll tests passed!")

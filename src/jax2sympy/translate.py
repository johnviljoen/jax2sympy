from jax.core import Literal
import sympy as sy
from jax2sympy.primitive_mapping import primitive_to_sympy_op
import numpy as np
import itertools as it
import functools as ft

def get_sym_invar(invars, var2sym):
    inexprs = []
    for inv in invars:
        if isinstance(inv, Literal):
            val = inv.val
            if isinstance(val, float) or isinstance(val, int) or isinstance(val, np.ndarray):
                inexprs.append(np.asarray(val))
            else:
                raise Exception("unknown jax core Literal encountered in jaxpr")
        else:
            inexprs.append(var2sym[inv])
    return inexprs

def get_sym_outvar(inexprs, eqn):
    prim_name = eqn.primitive.name
    if prim_name not in primitive_to_sympy_op:
        raise NotImplementedError(f"No Sympy translation rule for JAX primitive '{prim_name}'.")
    out_expr = primitive_to_sympy_op[prim_name](inexprs, eqn)
    return [out_expr]

def jaxpr_to_sympy_expressions(jaxpr, var2sym={}, x_cnt=0, c_cnt=0):
    
    def append_vars(vars, var2sym, sym_name, cnt):
        for i, var in enumerate(vars): # go through all inputs to the jaxpr
            assert var not in var2sym # double check we are not overwriting anything
            shape = var.aval.shape # the abstract shape of the input
            num_elements = np.prod(shape)
            symbols_list = sy.symbols([f'{sym_name}{i + cnt}_{j}' for j in range(num_elements)]) # create symbols for every element of the array
            symbolic_array = np.array(symbols_list).reshape(shape)
            var2sym[var] = symbolic_array
        return var2sym, i+1
    
    def substitute_symbol(symbol, sym_map):
        return symbol.subs(sym_map)

    var2sym, x_cnt = append_vars(jaxpr.invars, var2sym, 'x', cnt=x_cnt)
    if jaxpr.constvars != []:
        var2sym, c_cnt = append_vars(jaxpr.constvars, var2sym, 'c', cnt=c_cnt)

    for i, eqn in enumerate(jaxpr.eqns):
        print(f"eqn number: {i}")
        print(f"eqn: {eqn}")
        if eqn.primitive.name == "pjit":
            _jaxpr = eqn.params.get("jaxpr", None)
            sub_sym_out, sub_var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(_jaxpr.jaxpr, var2sym={}, x_cnt=x_cnt, c_cnt=c_cnt)
            assert len(sub_sym_out) == 1
            sub_sym_out = sub_sym_out[0]
            sym_in = get_sym_invar(eqn.invars, var2sym)
            assert len(sym_in) == 1
            sym_in = sym_in[0]
            sub_sym_in = sub_var2sym[next(iter(sub_var2sym))]
            sym_map = {key: value for key, value in zip(sub_sym_in.flat, sym_in.flat)}
            sub_sym_out = np.vectorize(substitute_symbol)(sub_sym_out, sym_map)
            exprs = [sub_sym_out]
        else:
            inexprs = get_sym_invar(eqn.invars, var2sym)
            exprs = get_sym_outvar(inexprs, eqn)
        for outvar, expr in zip(eqn.outvars, exprs):
            expr = np.asarray(expr) # check we dont lose the array with squeezing and indexing
            print(f"expression added to var2sym: {expr} under: {outvar}")
            assert outvar.aval.shape == expr.shape
            var2sym[outvar] = expr

    out_syms = [var2sym[outvar] for outvar in jaxpr.outvars]
    return out_syms, var2sym, x_cnt, c_cnt 

if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    from problems import mpc

    f, h, g, x0, gt, aux = mpc.quadcopter_nav(N=3) # N=100
    
    jaxpr = jax.make_jaxpr(h)(x0).jaxpr

    prims = []
    for i, eqn in enumerate(jaxpr.eqns):
        # print(i)
        prim = eqn.primitive.name
        if prim in prims:
            continue
        else:
            prims.append(prim)

        if prim != 'pjit':
            assert prim in primitive_to_sympy_op.keys()

    out_syms, var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(jaxpr)
    
    # assuming one vector input
    variables = var2sym[next(iter(var2sym))][0]
    indices = [int(x[3:]) for x in map(str, variables)]
    get_idx = {k: v for k, v in zip(variables, indices)}

    # build the sparse jacobian structure
    coords = []
    for i, output in enumerate(np.array(out_syms).flatten()):
        x_symbols = [s for s in output.free_symbols if str(s).startswith('x')]
        # differentiated_expr = [sy.diff(output, x) for x in x_symbols]
        row_coords = [[i, get_idx[s]] for s in x_symbols]
        coords = coords + row_coords
    coords = np.array(coords)
    
    nnz = coords.shape[0]

    # Plot the coordinates
    import matplotlib.pyplot as plt
    image_size = (len(indices), len(indices))
    image = np.zeros(image_size)

    # Fill the matrix with ones at the given coordinate positions
    for x, y in coords:
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Ensure indices are within bounds
            image[y, x] = 1

    # Plot the matrix using imshow
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='Greys', origin='upper')
    plt.title("Image Representation of Coordinates", fontsize=14)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.colorbar(label="Pixel Intensity")
    plt.savefig("test.png", dpi=500)

    print('fin')
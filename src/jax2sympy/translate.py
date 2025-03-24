from jax.core import Literal
import sympy as sy
from jax2sympy.primitive_mapping import primitive_to_sympy_op
import numpy as np

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
    if type(out_expr) != list: # multi var outputs are put in a list
        return [out_expr]
    else:
        return out_expr

def jaxpr_to_sympy_expressions(jaxpr, var2sym=dict(), x_cnt=0, c_cnt=0):
    
    def append_vars(vars, var2sym, sym_name, cnt):
        for i, var in enumerate(vars): # go through all inputs to the jaxpr
            assert var not in var2sym # double check we are not overwriting anything
            shape = var.aval.shape # the abstract shape of the input
            num_elements = int(np.prod(shape))
            symbols_list = sy.symbols([f'{sym_name}{i + cnt}_{j}' for j in range(num_elements)]) # create symbols for every element of the array
            symbolic_array = np.array(symbols_list).reshape(shape)
            var2sym[var] = symbolic_array
        return var2sym, i+1

    def substitute_symbol(symbol, sym_map):
        return symbol.subs(sym_map)

    var2sym, x_cnt = append_vars(jaxpr.jaxpr.invars, var2sym, 'x', cnt=x_cnt)
    if jaxpr.jaxpr.constvars != []:
        var2sym, c_cnt = append_vars(jaxpr.jaxpr.constvars, var2sym, 'c', cnt=c_cnt)

    # (OPTIONAL) sub in real values for the constants
    # def sub_consts(var2sym, consts):
    #     const_idx = 0
    #     for key, arr in list(var2sym.items()):
    #         # Ensure 'arr' is an array of object (sympy-like) and check if *all* entries start with 'c'
    #         if (isinstance(arr, np.ndarray) and arr.dtype == object
    #             and all(str(sym).startswith('c') for sym in arr.ravel())):
    #             var2sym[key] = consts[const_idx]
    #             const_idx += 1
    #     return var2sym
    # var2sym = sub_consts(var2sym, jaxpr.consts)

    for i, eqn in enumerate(jaxpr.eqns):
        # print(f"eqn number: {i}")
        # print(f"eqn: {eqn}")
        if eqn.primitive.name == "pjit":
            _jaxpr = eqn.params.get("jaxpr", None)
            sub_syms_out, sub_var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(_jaxpr, var2sym={}, x_cnt=x_cnt, c_cnt=c_cnt)
            syms_in = get_sym_invar(eqn.invars, var2sym)
            iterator = iter(sub_var2sym)
            sub_syms_in = [sub_var2sym[next(iterator)] for _ in range(len(syms_in))]
            sym_map = {}
            for sym_in, sub_sym_in in zip(syms_in, sub_syms_in):
                sym_map.update({key: value for key, value in zip(sub_sym_in.flat, sym_in.flat)})
            sub_syms_out = [np.vectorize(substitute_symbol)(f, sym_map) for f in sub_syms_out]
            exprs = sub_syms_out
        else:
            inexprs = get_sym_invar(eqn.invars, var2sym)
            exprs = get_sym_outvar(inexprs, eqn)
        exprs = [np.asarray(expr) for expr in exprs] # check we dont lose the array with squeezing and indexing
        for outvar, expr in zip(eqn.outvars, exprs):
            # print(f"expression added to var2sym: {expr} under: {outvar}")
            assert outvar.aval.shape == expr.shape
            var2sym[outvar] = expr

    out_syms = [var2sym[outvar] for outvar in jaxpr.jaxpr.outvars]
    return out_syms, var2sym, x_cnt, c_cnt 

if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    from problems import mpc

    f, h, g, x0, gt, aux = mpc.quadcopter_nav(N=3) # N=100
    
    jaxpr = jax.make_jaxpr(h)(x0)

    prims = []
    for i, eqn in enumerate(jaxpr.eqns):
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
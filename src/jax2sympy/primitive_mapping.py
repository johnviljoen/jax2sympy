import jax
import sympy as sy
import numpy as np
from jax.core import Literal

# elementwise
_sym_add = lambda a, b: a + b
_sym_sub = lambda a, b: a - b
_sym_mul = lambda a, b: a * b
_sym_div = lambda a, b: a / b
_sym_pow = lambda a, b: a ** b
_sym_integer_pow = lambda a, exp: a ** exp
_sym_neg = lambda a:  -a
_sym_exp = lambda a:  np.vectorize(sy.exp)(a)
_sym_log = lambda a:  np.vectorize(sy.log)(a)
_sym_sin = lambda a:  np.vectorize(sy.sin)(a)
_sym_cos = lambda a:  np.vectorize(sy.cos)(a)
_sym_tan = lambda a:  np.vectorize(sy.tan)(a)
_sym_asin = lambda a: np.vectorize(sy.asin)(a)
_sym_acos = lambda a: np.vectorize(sy.acos)(a)
_sym_atan = lambda a: np.vectorize(sy.atan)(a)
_sym_sinh = lambda a: np.vectorize(sy.sinh)(a)
_sym_cosh = lambda a: np.vectorize(sy.cosh)(a)
_sym_tanh = lambda a: np.vectorize(sy.tanh)(a)
_sym_sqrt = lambda a: np.vectorize(sy.sqrt)(a)
_sym_sign = lambda a: np.vectorize(sy.sign)(a)
_sym_eq = lambda a, b:  np.vectorize(lambda a, b: int(a == b))(a, b)
_sym_ne = lambda a, b:  np.vectorize(sy.Ne)(a, b)
_sym_lt = lambda a, b:  np.vectorize(sy.StrictLessThan)(a, b)
_sym_le = lambda a, b:  np.vectorize(sy.LessThan)(a, b)
_sym_gt = lambda a, b:  np.vectorize(sy.StrictGreaterThan)(a, b)
_sym_ge = lambda a, b:  np.vectorize(sy.GreaterThan)(a, b)
_sym_and = lambda a, b: np.vectorize(sy.And)(a, b)
_sym_or = lambda a, b:  np.vectorize(sy.Or)(a, b)
_sym_not = lambda a:    np.vectorize(sy.Not)(a)
_sym_max = lambda a, b: np.vectorize(sy.Max)(a, b)
_sym_min = lambda a, b: np.vectorize(sy.Min)(a, b)

# array
def _sym_reduce_sum(a, eqn):
    assert len(a) == 1
    return np.sum(a[0], axis=eqn.params["axes"])

# add any for our purposes is just add
def _sym_add_any(inexprs):
    # Use broadcasting to add arrays sequentially
    result = inexprs[0]
    for array in inexprs[1:]:
        result = np.add(result, array)  # np.add handles broadcasting
    return result

def _sym_select_n(inexprs):
    boolean = inexprs[0]
    true_arr = inexprs[1]
    false_arr = inexprs[2]
    return np.where(boolean, true_arr, false_arr)

def _sym_pad(inexprs, eqn):

    assert len(inexprs) == 2

    inexpr = inexprs[0]
    padding_value = inexprs[1]
    padding_config = eqn.params["padding_config"]

    # Create the padded shape
    padded_shape = [
        low + (inexpr.shape[i] - 1) * (interior + 1) + 1 + high
        for i, (low, high, interior) in enumerate(padding_config)
    ]

    # Create an array filled with the padding value
    result = np.full(padded_shape, padding_value, dtype=inexpr.dtype)
    
    # Calculate slices to place the original array into the padded result
    insert_slices = tuple(
        slice(low, low + inexpr.shape[i] * (interior + 1), interior + 1)
        for i, (low, high, interior) in enumerate(padding_config)
    )

    # Place the operand in the result array
    result[insert_slices] = inexpr
    return result

_sym_convert_element_type = lambda a: a

def _sym_reshape(a, new_sizes, dimensions):
    if dimensions is None:
        a = a.reshape(*new_sizes)
    else:
        raise NotImplementedError
    return a

def _sym_squeeze(a, dimensions):
    slices = [0 if i in dimensions else slice(None) for i in range(a.ndim)]
    return a[tuple(slices)]

# builds an array on device - this can be simplified
def _sym_iota(eqn):
    shape = eqn.params["shape"]
    dimension = eqn.params["dimension"]
    return np.arange(shape[dimension]).reshape(
        [1 if i != dimension else shape[dimension] for i in range(len(shape))]
    ).repeat(np.prod(shape) // shape[dimension], axis=dimension).reshape(shape)

def _sym_split(inexprs, eqn):
    print("WARNING: check functionality of split")
    sizes = eqn.params["sizes"]
    axis = eqn.params["axis"]
    assert len(inexprs) == 1
    operand = inexprs[0]
    indices = np.cumsum(sizes[:-1])  # Calculate split indices
    return np.split(operand, indices, axis=axis)

def _sym_slice(a, start_indices, limit_indices, strides):
    axis_slice = []
    if strides is not None:
        for start, limit, stride in zip(start_indices, limit_indices, strides):
            axis_slice.append([start, limit, stride])
    else:
        for start, limit in zip(start_indices, limit_indices):
            axis_slice.append([start, limit, 1])
    return a[tuple([slice(s[0], s[1], s[2]) for s in axis_slice])]

def _sym_dot_general(a, b, eqn):

    (contracting_dims_A, contracting_dims_B), (batch_dims_A, batch_dims_B) = eqn.params.get("dimension_numbers", None)

    a_ein = [chr(97 + i) for i in range(len(a.shape))]
    b_ein = [chr(97 + i) for i in range(len(a_ein), len(b.shape) + len(a_ein))]

    # align b_ein with a_ein
    b_ein[contracting_dims_B[0]] = a_ein[contracting_dims_A[0]]

    # form output shape - contracting dimension removed from larger dimension input
    if len(a.shape) >= len(b.shape):
        out_ein = a_ein[:contracting_dims_A[0]] + a_ein[contracting_dims_A[0] + 1:]
    else:
        out_ein = b_ein[:contracting_dims_B[0]] + b_ein[contracting_dims_B[0] + 1:]

    # form strings from lists of characters
    a_ein = ''.join(a_ein)
    b_ein = ''.join(b_ein)
    out_ein = ''.join(out_ein)

    return np.einsum(f"{a_ein},{b_ein}->{out_ein}", a, b)

def _sym_gather(operand, start_indices, dimension_numbers, slice_sizes, mode):
    
    # Unpack dimension numbers
    offset_dims = dimension_numbers.offset_dims
    # only can handle one offset dim rn
    assert len(offset_dims) == 1
    collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
    start_index_map = dimension_numbers.start_index_map

    # Handle index_vector_dim (assume last dimension if not specified)
    index_vector_dim = len(start_indices.shape) - 1

    # Initialize the output shape
    batch_dims = [dim for dim in range(len(start_indices.shape)) if dim != index_vector_dim]
    batch_shape = [start_indices.shape[d] for d in batch_dims]
    offset_shape = [slice_sizes[dim] for dim in range(len(slice_sizes)) if dim not in collapsed_slice_dims]
    output_shape = batch_shape + offset_shape
    output = np.empty(output_shape, dtype=operand.dtype)

    # Precompute mapping for remapped_offset_dims
    remapped_offset_dims = []
    operand_rank = len(operand.shape)
    assert operand_rank == 1
    available_dims = [dim for dim in range(operand_rank) if dim not in collapsed_slice_dims]
    assert len(offset_dims) == 1
    for idx, offset_dim in enumerate(offset_dims):
        remapped_offset_dims.append(available_dims[idx])

    # Iterate over all indices in the output array
    for out_index in np.ndindex(*output_shape):
        # Step 1: Extract batch indices G
        G = [out_index[d] for d in range(len(batch_dims))]

        # Step 2: Compute S using start_indices
        S = [0] * len(start_index_map)
        for i, map_dim in enumerate(start_index_map):
            combined_index = tuple(G[:index_vector_dim] + [i] + G[index_vector_dim:])
            S[i] = start_indices[combined_index]

        # Step 3: Compute Sin
        Sin = [0] * operand_rank
        for i, map_dim in enumerate(start_index_map):
            Sin[map_dim] = S[i]

        # Step 4: Compute Oin
        Oin = [0] * operand_rank
        for i, offset_dim in enumerate(offset_dims):
            Oin[remapped_offset_dims[i]] = out_index[len(batch_dims) + i]

        # NOTE MODIFIED: In = tuple(Sin[d] + Oin[d] for d in range(operand_rank))
        # Step 5: Compute In = Sin + Oin
        In = tuple(Sin[d] + Oin[d] for d in range(operand_rank))
        In = (In[0] - offset_dim * operand.shape[0],) # the offset

        # Step 6: Handle out-of-bounds indices
        if mode == jax.lax.GatherScatterMode.CLIP:
            In = tuple(max(0, min(idx, dim - 1)) for idx, dim in zip(In, operand.shape))
        elif mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            pass  # Assume indices are valid
        else:
            if any(idx < 0 or idx >= dim for idx, dim in zip(In, operand.shape)):
                raise IndexError(f"Out-of-bounds index: {In}")

        # Step 7: Populate the output array
        output[out_index] = operand[In]

    return output

def _sym_scatter(operand, scatter_indices, updates, dimension_numbers, mode):
    
    # Unpack dimension numbers
    update_window_dims = dimension_numbers.update_window_dims
    inserted_window_dims = dimension_numbers.inserted_window_dims
    scatter_dims_to_operand_dims = dimension_numbers.scatter_dims_to_operand_dims

    # Initialize output array with operand contents
    output = operand.copy()

    # Determine update_scatter_dims: dimensions in updates not used for the window
    update_scatter_dims = [i for i in range(updates.ndim) if i not in update_window_dims]

    def build_window_dims_to_operand_dims(update_window_dims, inserted_window_dims, operand_rank):
        """
        Build mapping from update_window_dims to operand dimensions, excluding inserted_window_dims.
        """
        # Get all operand dimensions excluding inserted_window_dims
        available_dims = [i for i in range(operand_rank) if i not in inserted_window_dims]
        mapping = {dim: available_dims[idx] for idx, dim in enumerate(update_window_dims)}
        return mapping

    for update_index in np.ndindex(*updates.shape):
        print(update_index)
        # Step 1: Extract G from update_index
        G = [update_index[dim] for dim in update_scatter_dims]

        # Step 2: Compute S using scatter_indices and Combine(G, i)
        S = [0] * len(scatter_dims_to_operand_dims)
        for i, scatter_dim in enumerate(scatter_dims_to_operand_dims):
            # Combine G and scatter_indices
            combined = tuple(G[:scatter_dim] + [i] + G[scatter_dim:])
            S[i] = scatter_indices[combined]

        # Step 3: Compute Sin
        Sin = [0] * operand.ndim
        for i, scatter_dim in enumerate(scatter_dims_to_operand_dims):
            Sin[scatter_dim] = S[i]

        # Step 4: Compute Win
        Win = [0] * operand.ndim
        window_mapping = build_window_dims_to_operand_dims(update_window_dims, inserted_window_dims, operand.ndim)
        for i in update_window_dims:
            Win[window_mapping[i]] = update_index[i]

        # Step 5: Calculate final index I = Win + Sin
        I = tuple(int(Win[d] + Sin[d]) for d in range(operand.ndim))

        # Step 6: Handle out-of-bounds indices
        if mode == jax.lax.GatherScatterMode.CLIP:
            I = tuple(max(0, min(idx, dim - 1)) for idx, dim in zip(I, operand.shape))
        elif mode == jax.lax.GatherScatterMode.FILL_OR_DROP:
            if any(idx < 0 or idx >= dim for idx, dim in zip(I, operand.shape)):
                continue  # Skip out-of-bounds updates
        elif mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            pass  # Assume all indices are in bounds
        else:
            if any(idx < 0 or idx >= dim for idx, dim in zip(I, operand.shape)):
                raise IndexError(f"Out-of-bounds index: {I}")

        # Step 7: Apply the update
        output[I] = updates[update_index]

    return output

def _sym_broadcast_in_dim(a, shape, broadcast_dimensions):
    assert len(a) == 1
    a = a[0]
    reshaped_shape = [1] * len(shape)
    for i, dim in enumerate(broadcast_dimensions):
        reshaped_shape[dim] = a.shape[i]
    a_reshaped = np.reshape(a, reshaped_shape)
    return np.broadcast_to(a_reshaped, shape)

def _sym_concatenate(arr_list, dimension=0):
    return np.concat(arr_list, axis=dimension)

primitive_to_sympy_op = {
    "add":  lambda inexprs, eqn: _sym_add(*inexprs),
    "sub":  lambda inexprs, eqn: _sym_sub(*inexprs),
    "mul":  lambda inexprs, eqn: _sym_mul(*inexprs),
    "div":  lambda inexprs, eqn: _sym_div(*inexprs),
    "neg":  lambda inexprs, eqn: _sym_neg(*inexprs),
    "pow":  lambda inexprs, eqn: _sym_pow(*inexprs),
    "integer_pow": lambda inexprs, eqn: _sym_integer_pow(inexprs[0], eqn.params['y']),
    "sign": lambda inexprs, eqn: _sym_sign(*inexprs),
    "exp":  lambda inexprs, eqn: _sym_exp(*inexprs),
    "log":  lambda inexprs, eqn: _sym_log(*inexprs),
    "sin":  lambda inexprs, eqn: _sym_sin(*inexprs),
    "cos":  lambda inexprs, eqn: _sym_cos(*inexprs),
    "tan":  lambda inexprs, eqn: _sym_tan(*inexprs),
    "asin": lambda inexprs, eqn: _sym_asin(*inexprs),
    "acos": lambda inexprs, eqn: _sym_acos(*inexprs),
    "atan": lambda inexprs, eqn: _sym_atan(*inexprs),
    "sinh": lambda inexprs, eqn: _sym_sinh(*inexprs),
    "cosh": lambda inexprs, eqn: _sym_cosh(*inexprs),
    "tanh": lambda inexprs, eqn: _sym_tanh(*inexprs),
    "sqrt": lambda inexprs, eqn: _sym_sqrt(*inexprs),
    "eq":  lambda inexprs, eqn: _sym_eq(*inexprs),
    "ne":  lambda inexprs, eqn: _sym_ne(*inexprs),
    "lt":  lambda inexprs, eqn: _sym_lt(*inexprs),
    "le":  lambda inexprs, eqn: _sym_le(*inexprs),
    "gt":  lambda inexprs, eqn: _sym_gt(*inexprs),
    "ge":  lambda inexprs, eqn: _sym_ge(*inexprs),
    "not": lambda inexprs, eqn: _sym_not(*inexprs),
    "and": lambda inexprs, eqn: _sym_and(*inexprs),
    "or":  lambda inexprs, eqn: _sym_or(*inexprs),
    "max": lambda inexprs, eqn: _sym_max(*inexprs),
    "min": lambda inexprs, eqn: _sym_min(*inexprs),
    "convert_element_type": lambda inexprs, eqn: _sym_convert_element_type(inexprs),
    "broadcast_in_dim": lambda inexprs, eqn: _sym_broadcast_in_dim(
        inexprs,
        shape = eqn.params["shape"],
        broadcast_dimensions = eqn.params["broadcast_dimensions"],
    ),
    "concatenate": lambda inexprs, eqn: _sym_concatenate(inexprs, eqn.params["dimension"]),
    "slice":       lambda inexprs, eqn: _sym_slice(
        *inexprs,
        eqn.params["start_indices"],
        eqn.params["limit_indices"],
        eqn.params.get("strides", None),
    ),
    "squeeze":     lambda inexprs, eqn: _sym_squeeze(*inexprs, eqn.params["dimensions"]),
    "scatter":     lambda inexprs, eqn: _sym_scatter(
        *inexprs, 
        dimension_numbers = eqn.params["dimension_numbers"], 
        mode = eqn.params["mode"],
    ),
    "gather":      lambda inexprs, eqn: _sym_gather(
        *inexprs,
        dimension_numbers = eqn.params["dimension_numbers"],
        slice_sizes = eqn.params["slice_sizes"],
        mode = eqn.params["mode"],
    ),
    "add_any": lambda inexprs, eqn: _sym_add_any(inexprs),
    "pad": lambda inexprs, eqn: _sym_pad(inexprs, eqn),
    "select_n": lambda inexprs, eqn: _sym_select_n(inexprs),
    "reduce_sum": lambda inexprs, eqn: _sym_reduce_sum(inexprs, eqn),
    "iota": lambda inexprs, eqn: _sym_iota(eqn),
    "split": lambda inexprs, eqn: _sym_split(inexprs, eqn),
    "convert_element_type": lambda inexprs, eqn: _sym_convert_element_type(inexprs[0]),
    "dot_general": lambda inexprs, eqn: _sym_dot_general(inexprs[0], inexprs[1], eqn),
    "transpose": lambda inexprs, eqn: _sym_transpose(inexprs[0], eqn),
    "reshape": lambda inexprs, eqn: _sym_reshape(
        *inexprs,                # the expression being reshaped
        eqn.params["new_sizes"],
        eqn.params["dimensions"],
    ),
}

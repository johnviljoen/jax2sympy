# Jax2SymPy

This repo translates jaxpr graphs into a SymPy elementwise expressions. This allows for symbolic analysis of the system. 

My use case is finding sparsity pattern of jacobians and hessians of vector valued functions through symbolic differentiation, find example code under sparsity.py. 

I built this repo as part of a project to GPU-batch solve many IPOPT optimizations in [jaxipm](https://github.com/johnviljoen/jaxipm).

# Installation

pip:
```bash
pip install jax2sympy
```
uv:
```bash
uv pip install jax2sympy
```

> Using a uv-managed project instead? Just `uv add jax2sympy`.


# Usage

Get the symbolic Jacobian of a JAX function together with its sparsity pattern:

```python
import jax.numpy as jnp
from jax2sympy import get_symbolic_jacobian

def f(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[2]), x[0] + x[2]])

x = jnp.ones(3)

# sym_jac_val: SymPy expressions for the non-zero Jacobian entries
# sym_jac_coo: (nnz, 2) array of [output_index, input_index] coordinates
sym_jac_val, sym_jac_coo, input_symbols, const_symbols = get_symbolic_jacobian(f, x)

print(sym_jac_coo)  # sparsity pattern
print(sym_jac_val)  # symbolic derivatives
```

`get_symbolic_hessian` returns the analogous result for second derivatives, with
`(nnz, 3)` coordinates of `[output_index, i, j]`.

# Public API

- `jaxpr_to_sympy_expressions` — translate a `jaxpr` into SymPy expressions
- `get_symbolic_jacobian` / `get_symbolic_hessian` — symbolic derivatives + sparsity
- `sparse_jacobian_sym` / `sparse_hessian_sym` — build sparse symbolic derivative functions
- `get_sparsity_pattern`, `sparse_jacobian`, `sparse_hessian`, `get_dense` — numeric sparsity utilities

# Citation

```
@article{viljoen2026scaling,
  title={Scaling Nonlinear Optimization: Many Problems One GPU},
  author={Viljoen, John and Haffner, Johanna and Tomizuka, Masayoshi and Mehr, Negar},
  journal={arXiv preprint arXiv:2606.26341},
  year={2026}
}
```
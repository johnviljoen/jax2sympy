"""Regression: the jaxipm-facing surface works across jax versions.

jax >= 0.9 emits a first-class 'stack' primitive (jnp.stack no longer lowers
to concatenate+reshape as on <= 0.8); this test drives an OCP-shaped problem
whose dynamics open with jnp.stack — quaternion products, trig, drag terms —
through get_sparsity_pattern + sparse_jacobian_sym + sparse_hessian_sym and
checks numeric parity against dense jax autodiff. Passes identically on
jax 0.8.0 (where 'stack' never appears) and jax >= 0.9.

Run: pytest tests/ -q   (requires sympy2jax)
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax2sympy.sparsify import get_sparsity_pattern
from jax2sympy.sparsify_sym import sparse_jacobian_sym, sparse_hessian_sym

# problem dimensions: N states of size nx, N-1 inputs of size nu, flat z
N, nx, nu, Ts = 4, 7, 2, 0.1
nz = N * nx + (N - 1) * nu


def dyn(x, u):
    """Quadcopter-flavored dynamics: jnp.stack of quaternion products, trig
    and quadratic drag — the shapes of expression jaxipm problems produce."""
    return jnp.stack([
        x[4],
        -0.5 * x[2] * x[3] + 0.5 * x[1] * x[5],
        0.5 * x[1] * x[3] + jnp.sin(x[6]) * u[0],
        jnp.cos(x[2]) * u[1] - jnp.sign(x[4]) * x[4] ** 2,
        (x[1] * x[2] - x[3] * x[6]) * u[0],
        jnp.exp(-x[5]) + x[0] * u[1],
        jnp.sqrt(x[6] ** 2 + 1.0) - 1.0,
    ])


def z_to_xu(z):
    return z[: N * nx].reshape(N, nx), z[N * nx:].reshape(N - 1, nu)


xr = jnp.tile(jnp.linspace(0.1, 0.7, nx), (N, 1))
x0c = xr[0]


def f(z):  # tracking objective (scalar)
    x, u = z_to_xu(z)
    return jnp.sum((x - xr) ** 2) + 1e-3 * jnp.sum(u ** 2)


def c(z):  # initial condition + Euler dynamics defects
    x, u = z_to_xu(z)
    cons = [x[0] - x0c]
    for k in range(N - 1):
        cons.append(x[k + 1] - (x[k] + Ts * dyn(x[k], u[k])))
    return jnp.concatenate(cons).flatten()


def d(z):  # obstacle-style inequalities
    x, u = z_to_xu(z)
    return (x[:, 0] - 1.0) ** 2 + (x[:, 1] - 1.0) ** 2 - 0.25


_rng = np.random.default_rng(0)
_z0 = jnp.asarray(np.concatenate(
    [np.tile(np.asarray(xr[0]), N), _rng.uniform(0.5, 1.5, (N - 1) * nu)]))


def _check(fn):
    jc = jnp.array(get_sparsity_pattern(fn, _z0, type="jacobian"),
                   dtype=jnp.int32)
    hc = jnp.array(get_sparsity_pattern(fn, _z0, type="hessian"),
                   dtype=jnp.int32)
    m = int(np.atleast_1d(np.asarray(fn(_z0))).size)
    jfn = sparse_jacobian_sym(fn, _z0, coo_pattern=jc, out_shape=(m, nz))
    hfn = sparse_hessian_sym(fn, _z0, coo_pattern=hc, out_shape=(m, nz, nz))
    dense_j = np.asarray(jax.jacobian(fn)(_z0)).reshape(m, nz)
    np.testing.assert_allclose(
        np.asarray(jfn(_z0).todense()).reshape(m, nz), dense_j, atol=1e-10)
    dense_h = np.asarray(jax.hessian(fn)(_z0)).reshape(m, nz, nz)
    np.testing.assert_allclose(
        np.asarray(hfn(_z0).todense()).reshape(m, nz, nz), dense_h,
        atol=1e-10)


def test_objective_jac_hess():
    _check(f)


def test_dynamics_constraints_jac_hess():
    # dyn() opens with jnp.stack -> exercises the 'stack' rule on jax >= 0.9
    _check(c)


def test_inequality_jac_hess():
    _check(d)


if __name__ == "__main__":
    test_objective_jac_hess()
    test_dynamics_constraints_jac_hess()
    test_inequality_jac_hess()
    print(f"jax {jax.__version__}: jaxipm-facing surface OK")

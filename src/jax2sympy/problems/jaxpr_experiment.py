import jax
import jax.numpy as jnp

def func(x, y, z, a, b, c):
    return jnp.float32([jnp.sin(x), y, jnp.exp(z)])

x = y = z = a = b = c = jnp.float32(2.0)

# print(jax.make_jaxpr(jax.jacobian(func))(x, y, z, a, b, c))

def func(x):
    return jnp.sin(x + 1)

print(jax.make_jaxpr(func)(x))
print(jax.make_jaxpr(jax.grad(func))(x))

pass
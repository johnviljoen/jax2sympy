import jax
import jax.numpy as jnp

def func(x, y, z, a, b, c):
    return jnp.float32([jnp.sin(x), y, jnp.exp(z)])

x = y = z = a = b = c = jnp.float32(2.0)

print(jax.make_jaxpr(jax.jacobian(func, argnums=jnp.arange(6)))(x, y, z, a, b, c))



pass
import jax
import jax.numpy as jnp
from network_parameters import cfl, x, T

dx = x[1] - x[0]
dt_max = 0.005
n_steps = int(T / dt_max) + 10

@jax.jit
def advection_solver(u):
    def step(carry, _):
        u, F, t = carry
        a = jnp.sin(t)
        dt = jnp.minimum(cfl * dx / (jnp.abs(a) + 1e-10), dt_max)
        f_face = jnp.where(a > 0, a * u, a * jnp.roll(u, -1))
        F = F + f_face
        u = u - dt / dx * (f_face - jnp.roll(f_face, 1))
        return (u, F, t + dt), None

    F_init = jnp.zeros_like(u)
    (_, F_target, _), _ = jax.lax.scan(step, (u, F_init, 0.0), None, length=n_steps)
    return F_target

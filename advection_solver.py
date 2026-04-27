import jax
import jax.numpy as jnp
from network_parameters import cfl, x, T

dx = x[1] - x[0]
dt_max = 0.005
n_steps = 1000

def advection_solver(u, n_steps):
  t = 0
  dx = x[1] - x[0]
  eps = 1e-10
  F = jnp.zeros_like(u)
  u0 = u
  for t in range(n_steps):
    a = 1
    dt = jnp.minimum(cfl * dx / jnp.max(jnp.abs(a)), 0.005)
    f_face = jnp.where(a > 0, 
                  dt * a * u,
                  dt * a * jnp.roll(u, -1))
    u = u - 1/dx * (f_face - jnp.roll(f_face, 1))
    F += f_face
    t += dt
  return F

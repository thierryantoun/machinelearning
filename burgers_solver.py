import functools
import jax
import jax.numpy as jnp
from network_parameters import cfl, x, n_steps

dx = x[1] - x[0]

def flux(u_L, u_R):
    s_L = jnp.minimum(u_L, u_R)
    s_R = jnp.maximum(u_L, u_R)
    return jnp.where(s_L >= 0, 0.5 * u_L**2,
                     jnp.where(s_R <= 0, 0.5 * u_R**2,
                               (s_R * 0.5 * u_L**2 - s_L * 0.5 * u_R**2 + s_L * s_R * (u_R - u_L)) / (s_R - s_L)))


@functools.partial(jax.jit, static_argnums=(1,))
def burgers_solver(u, n_steps):
    def step(carry, _):
        u, t, F = carry
        dt_step = cfl * dx / (jnp.max(jnp.abs(u)) + 1e-10)
        f_face = flux(u, jnp.roll(u, -1))
        u = u - dt_step / dx * (f_face - jnp.roll(f_face, 1))
        F = F + dt_step * f_face
        t = t + dt_step
        return (u, t, F), None

    (u_final, t_final, F_final), _ = jax.lax.scan(
        step, (u, 0.0, jnp.zeros_like(u)), None, length=n_steps
    )
    return u_final, F_final / t_final, t_final

# u0 = jnp.sin(2 * jnp.pi * x)
# u_final, F_final, t_final = burgers_solver(u0, n_steps)

# import matplotlib.pyplot as plt
# plt.plot(x, u0, label='u0')
# plt.plot(x, u_final, label='u_final')
# plt.title('Résultat du solveur de Burgers')
# plt.xlabel('x')
# plt.ylabel('u')
# plt.legend()
# plt.show()
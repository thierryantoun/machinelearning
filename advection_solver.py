import functools
import jax
import jax.numpy as jnp
from network_parameters import cfl, x, a

dx = x[1] - x[0]
dt_max = 0.005
n_steps = 10

@functools.partial(jax.jit, static_argnums=(1,))
def advection_solver(u, n_steps):
    dx = x[1] - x[0]
    epsilon = 1e-10

    def step(carry, _):
        u, t, F = carry
        dt = jnp.minimum(cfl * dx / (jnp.abs(a) + epsilon), 0.005)
        f_face = jnp.where(a > 0,
                           dt * a * u,
                           dt * a * jnp.roll(u, -1))
        u = u - 1/dx * (f_face - jnp.roll(f_face, 1))
        F += f_face
        t += dt
        return (u, t, F), None

    (u_final, t_final, F_final), _ = jax.lax.scan(
        step,
        (u, 0.0, jnp.zeros_like(u)),
        None,
        length=n_steps
    )
    return F_final, t_final

# import matplotlib.pyplot as plt
# u0 = jnp.sin(2 * jnp.pi * x)
# F, _ = advection_solver(u0, n_steps)
# u = u0 - 1/dx * (F - jnp.roll(F, 1))
# plt.plot(x, u0, label='u0')
# plt.plot(x, u, label='u')
# plt.legend()
# plt.title('Résultat du solveur d\'advection')
# plt.xlabel('x')
# plt.ylabel('u')
# plt.show()
# plt.savefig("advection_solver_test.png")
import jax
import jax.numpy as jnp
from network_parameters import cfl

gamma = 5. / 3.

# U : (nvar, n) variables conservatives (rho, rho*u, E), l'espace reste sur le dernier axe
nvar = 3
ID = 0
IU = 1
IE = 2


def primitives(U):
    rho = U[ID]
    u   = U[IU] / rho
    p   = (U[IE] - 0.5 * u**2 * rho) * (gamma - 1.0)
    return rho, u, p


# flux à la face entre U_L et U_R, (nvar, n)
def flux(U_L, U_R):
    rho_L, u_L, p_L = primitives(U_L)
    rho_R, u_R, p_R = primitives(U_R)
    a_L = rho_L * jnp.sqrt(gamma * p_L / rho_L)
    a_R = rho_R * jnp.sqrt(gamma * p_R / rho_R)

    aface = 1.1 * jnp.maximum(a_L, a_R)

    ustar = 0.5 * (u_L + u_R) - 0.5 * (p_R - p_L) / aface
    theta = jnp.minimum(jnp.abs(ustar) / jnp.maximum(a_L / rho_L, a_R / rho_R), 1.0)
    pstar = 0.5 * (p_L + p_R) - 0.5 * (u_R - u_L) * aface * theta

    F = ustar * jnp.where(ustar > 0, U_L, U_R)
    return F.at[IU].add(pstar).at[IE].add(pstar * ustar)


@jax.jit
def euler_solver(U, T_target):
    dx = 1.0 / U.shape[-1]   # domaine périodique [0, 1)

    def cond_fn(carry):
        U, t, F = carry
        return t < T_target

    def body_fn(carry):
        U, t, F = carry
        rho, u, p = primitives(U)
        dt_cfl = cfl * dx / jnp.max(jnp.abs(u) + jnp.sqrt(gamma * p / rho))
        dt_step = jnp.minimum(dt_cfl, T_target - t)  # ne jamais depasser T_target
        f_face = flux(U, jnp.roll(U, -1, axis=-1))
        U_new = U - dt_step / dx * (f_face - jnp.roll(f_face, 1, axis=-1))
        F_new = F + dt_step * f_face
        t_new = t + dt_step
        return (U_new, t_new, F_new)

    U_final, t_final, F_final = jax.lax.while_loop(
        cond_fn, body_fn, (U, 0.0, jnp.zeros_like(U))
    )
    return U_final, F_final / T_target, T_target

from functools import partial

import jax
import jax.numpy as jnp
from network_parameters import cfl

gamma = 5. / 3.
K_A = 1.1   # aface = K_A * max(rho_L c_L, rho_R c_R), partagé par flux et pas de temps

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

    aface = K_A * jnp.maximum(a_L, a_R)

    ustar = 0.5 * (u_L + u_R) - 0.5 * (p_R - p_L) / aface
    # pas de correction bas-Mach (theta) : elle supprimait la dissipation acoustique
    # et créait des oscillations (damier, dépassements aux chocs)
    pstar = 0.5 * (p_L + p_R) - 0.5 * (u_R - u_L) * aface

    F = ustar * jnp.where(ustar > 0, U_L, U_R)
    return F.at[IU].add(pstar).at[IE].add(pstar * ustar)


# return_dist=True : renvoie aussi dist (2,) = somme_k s_k dt_k / dx, nombre de cellules parcourues par l'onde la plus
# rapide pendant T_target : dist[0] avec la vitesse CFL s_k = max(|u| + s_ac) (vaut cfl par pas complet, majorant),
# dist[1] avec la vitesse physique max(|u| + c) (cône physique).
# Statique : change le nombre de sorties, donc une compilation par valeur
@partial(jax.jit, static_argnames="return_dist")
def euler_solver(U, T_target, return_dist=False):
    dx = 1.0 / U.shape[-1]   # domaine périodique [0, 1)

    def cond_fn(carry):
        U, t, F, dist = carry
        return t < T_target

    def body_fn(carry):
        U, t, F, dist = carry
        rho, u, p = primitives(U)
        # CFL acoustique : la cellule i voit aface_{i-1/2} et aface_{i+1/2} (max des impédances
        # voisines), donc vitesse effective max(aface) / rho_i, plus grande que c_i près d'un saut
        a = rho * jnp.sqrt(gamma * p / rho)
        aface = K_A * jnp.maximum(a, jnp.roll(a, -1, axis=-1))
        s_ac = jnp.maximum(aface, jnp.roll(aface, 1, axis=-1)) / rho
        s_max = jnp.max(jnp.abs(u) + s_ac)
        dt_cfl = cfl * dx / s_max
        dt_step = jnp.minimum(dt_cfl, T_target - t)  # ne jamais depasser T_target
        f_face = flux(U, jnp.roll(U, -1, axis=-1))
        U_new = U - dt_step / dx * (f_face - jnp.roll(f_face, 1, axis=-1))
        F_new = F + dt_step * f_face
        t_new = t + dt_step
        s_phys = jnp.max(jnp.abs(u) + jnp.sqrt(gamma * p / rho))   # cône physique, <= s_max
        return (U_new, t_new, F_new, dist + jnp.stack([s_max, s_phys]) * dt_step / dx)

    U_final, t_final, F_final, dist = jax.lax.while_loop(
        cond_fn, body_fn, (U, 0.0, jnp.zeros_like(U), jnp.zeros(2, U.dtype))
    )
    if return_dist:
        return U_final, F_final / T_target, T_target, dist
    return U_final, F_final / T_target, T_target

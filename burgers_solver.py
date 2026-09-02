import jax
import jax.numpy as jnp
from network_parameters import x, cfl

dx = x[1] - x[0]


def flux(u_L, u_R):
    s_L = jnp.minimum(u_L, u_R)
    s_R = jnp.maximum(u_L, u_R)
    return jnp.where(
        s_L >= 0, 0.5 * u_L**2,
        jnp.where(
            s_R <= 0, 0.5 * u_R**2,
            (s_R * 0.5 * u_L**2 - s_L * 0.5 * u_R**2 + s_L * s_R * (u_R - u_L)) / (s_R - s_L)
        )
    )


@jax.jit
def burgers_solver(u, T_target):
    """
    Integre Burgers jusqu'au temps physique T_target (pas un nombre de pas fixe).
    dt reste adaptatif (CFL) a chaque sous-pas ; le dernier sous-pas est tronque
    pour ne jamais depasser T_target. Tous les echantillons generes avec le meme
    T_target auront donc exactement le meme T -- plus d'heterogeneite T_min/T_max.

    Retourne (u_final, F_moyen, T_target) -- T_target est renvoye tel quel
    (au lieu du t_final accumule) pour que l'appelant recupere une valeur
    exacte et deterministe, sans erreur d'arrondi de sommation de dt.
    """

    def cond_fn(carry):
        u, t, F = carry
        return t < T_target

    def body_fn(carry):
        u, t, F = carry
        dt_cfl = cfl * dx / (jnp.max(jnp.abs(u)) + 1e-10)
        dt_step = jnp.minimum(dt_cfl, T_target - t)  # ne jamais depasser T_target
        f_face = flux(u, jnp.roll(u, -1))
        u_new = u - dt_step / dx * (f_face - jnp.roll(f_face, 1))
        F_new = F + dt_step * f_face
        t_new = t + dt_step
        return (u_new, t_new, F_new)

    u_final, t_final, F_final = jax.lax.while_loop(
        cond_fn, body_fn, (u, 0.0, jnp.zeros_like(u))
    )
    return u_final, F_final / T_target, T_target

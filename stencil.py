import math

import jax.numpy as jnp


# me donne le nombre de points à gauche et à droite du point central pour le stencil
def stencil_size(a_min, a_max, lam, margin=0):
    m_g = max(1, math.ceil(max(a_max, 0.0) * lam)) + margin
    m_d = max(1, math.ceil(max(-a_min, 0.0) * lam)) + margin
    return m_g - 1, m_d

# me donne les valeurs du stencil pour un vecteur u, avec p points à gauche et q points à droite
def stencil(u, p, q):
    return jnp.stack([jnp.roll(u, -i) for i in range(-p, q + 1)], axis=-1)

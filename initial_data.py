import jax
import jax.numpy as jnp
from jax import random
from network_parameters import K, x


def make_sinus_u0(key):
    k1, k2 = random.split(key)
    a_k = random.uniform(k1, (K,), minval=-1.0, maxval=1.0)
    phase_k = random.uniform(k2, (K,), minval=0.0, maxval=2 * jnp.pi)
    ks = jnp.arange(1, K + 1)
    u = jnp.sum(a_k[:, None] * jnp.sin(2 * jnp.pi * ks[:, None] * x[None, :] + phase_k[:, None]), axis=0)
    return u / jnp.max(jnp.abs(u))


def generate_initial_data(key, nb_frequences=K, x=x):
    key, subkey = random.split(key)
    # 0: sinus, 1: gaussiennes, 2: polynomes, 3: constante, 4: rampe,
    # 5: marches (constante par morceaux, discontinue), 6: creneau (pulse rectangulaire, discontinu)
    # 7: riemann (une seule discontinuite, deux niveaux aleatoires)
    # 8: sinus_hf (sinus multi-frequences, uniquement k > K, hors bande d'entrainement basse frequence)
    ic_type = random.randint(subkey, (), 0, 9)

    key, subkey = random.split(key)

    def make_sinus(subkey):
        a_k = random.uniform(subkey, (K,), minval=-1.0, maxval=1.0)
        key2, subkey2 = random.split(subkey)
        phase_k = random.uniform(subkey2, (K,), minval=0.0, maxval=2*jnp.pi)
        ks = jnp.arange(1, K+1)
        return jnp.sum(a_k[:, None] * jnp.sin(2*jnp.pi*ks[:, None]*x[None, :] + phase_k[:, None]), axis=0)

    def make_sinus_hf(subkey):
        "Sinus multi-frequences, uniquement des frequences hautes k > K = 40 (hors bande basse frequence du dataset d'entrainement)."
        a_k = random.uniform(subkey, (K,), minval=-1.0, maxval=1.0)
        key2, subkey2 = random.split(subkey)
        phase_k = random.uniform(subkey2, (K,), minval=0.0, maxval=2*jnp.pi)
        ks = jnp.arange(K + 1, 2 * K + 1)
        return jnp.sum(a_k[:, None] * jnp.sin(2*jnp.pi*ks[:, None]*x[None, :] + phase_k[:, None]), axis=0)

    def make_gaussiennes(subkey):
        n_gaussians = 4
        k1, k2, k3 = random.split(subkey, 3)
        centers = random.uniform(k1, (n_gaussians,), minval=0.0, maxval=1.0)
        widths  = random.uniform(k2, (n_gaussians,), minval=0.02, maxval=0.15)
        amps    = random.uniform(k3, (n_gaussians,), minval=-1.0, maxval=1.0)
        return jnp.sum(amps[:, None] * jnp.exp(-((x[None, :] - centers[:, None])**2) / (2 * widths[:, None]**2)), axis=0)

    def make_polynomes(subkey):
        k1, k2 = random.split(subkey)
        degree = 5
        coeffs = random.uniform(k1, (degree+1,), minval=-5.0, maxval=5.0)
        u = jnp.polyval(coeffs, x)
        u = u - (u[-1] - u[0]) * x - u[0]
        return u

    def make_constante(subkey):
        c = random.uniform(subkey, (), minval=-5.0, maxval=5.0)
        return jnp.ones_like(x) * c

    def make_rampe(subkey):
        k1, k2, k3, k4 = random.split(subkey, 4)
        center    = random.uniform(k1, (), minval=0.0, maxval=1.0)
        steepness = random.uniform(k2, (), minval=20.0, maxval=400.0)
        amp       = random.uniform(k3, (), minval=-1.0, maxval=1.0)
        sign      = jnp.sign(random.uniform(k4, (), minval=-1.0, maxval=1.0))
        d = x - center
        d = d - jnp.round(d)
        return amp * jnp.tanh(sign * steepness * d)

    def make_marches(subkey):
        "Fonction constante par morceaux : discontinuites de type Riemann multiples (periodique)."
        k1, k2 = random.split(subkey)
        n_seg = 5
        edges  = jnp.sort(random.uniform(k1, (n_seg - 1,), minval=0.0, maxval=1.0))
        levels = random.uniform(k2, (n_seg,), minval=-1.0, maxval=1.0)
        seg = jnp.sum(x[:, None] >= edges[None, :], axis=1)  # indice de segment pour chaque x
        return levels[seg]

    def make_creneau(subkey):
        "Pulse rectangulaire : deux discontinuites (double probleme de Riemann)."
        k1, k2, k3 = random.split(subkey, 3)
        center = random.uniform(k1, (), minval=0.2, maxval=0.8)
        width  = random.uniform(k2, (), minval=0.05, maxval=0.4)
        amp    = random.uniform(k3, (), minval=-1.0, maxval=1.0)
        return amp * (jnp.abs(x - center) < 0.5 * width).astype(x.dtype)

    def make_riemann(subkey):
        "Probleme de Riemann simple : une seule discontinuite, deux niveaux constants aleatoires."
        k1, k2, k3 = random.split(subkey, 3)
        split  = random.uniform(k1, (), minval=0.0, maxval=1.0)
        levelL = random.uniform(k2, (), minval=-1.0, maxval=1.0)
        levelR = random.uniform(k3, (), minval=-1.0, maxval=1.0)
        return jnp.where(x < split, levelL, levelR)

    is_constante = (ic_type == 3)
    u = jax.lax.switch(
        ic_type,
        [make_sinus, make_gaussiennes, make_polynomes, make_constante, make_rampe,
         make_marches, make_creneau, make_riemann, make_sinus_hf],
        subkey,
    )
    u = jnp.where(is_constante, u, u / jnp.max(jnp.abs(u)))

    return u
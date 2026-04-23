import jax
import jax.numpy as jnp
from jax import random
from network_parameters import K, x

def generate_initial_data(key, nb_frequences=K):
    key, subkey = random.split(key)
    ic_type = random.randint(subkey, (), 0, 3)  # 0: sinus, 1: gaussiennes, 2: polynomes

    key, subkey = random.split(key)

    def make_sinus(subkey):
        a_k = random.uniform(subkey, (K,), minval=-1.0, maxval=1.0)
        key2, subkey2 = random.split(subkey)
        phase_k = random.uniform(subkey2, (K,), minval=0.0, maxval=2*jnp.pi)
        ks = jnp.arange(1, K+1)
        return jnp.sum(a_k[:, None] * jnp.sin(2*jnp.pi*ks[:, None]*x[None, :] + phase_k[:, None]), axis=0)

    def make_gaussiennes(subkey):
        # somme de gaussiennes aléatoires
        n_gaussians = 4
        k1, k2, k3 = random.split(subkey, 3)
        centers = random.uniform(k1, (n_gaussians,), minval=0.0, maxval=1.0)
        widths  = random.uniform(k2, (n_gaussians,), minval=0.02, maxval=0.15)
        amps    = random.uniform(k3, (n_gaussians,), minval=-1.0, maxval=1.0)
        return jnp.sum(amps[:, None] * jnp.exp(-((x[None, :] - centers[:, None])**2) / (2 * widths[:, None]**2)), axis=0)

    def make_polynomes(subkey):
        # polynome aléatoire de degré 5 evalué sur [0,1], rendu périodique
        k1, k2 = random.split(subkey)
        degree = 5
        coeffs = random.uniform(k1, (degree+1,), minval=-1.0, maxval=1.0)
        u = jnp.polyval(coeffs, x)
        # rend périodique en soustrayant la droite reliant u(0) à u(1)
        u = u - (u[-1] - u[0]) * x - u[0]
        return u

    u = jax.lax.switch(ic_type, [make_sinus, make_gaussiennes, make_polynomes], subkey)
    u = u / jnp.max(jnp.abs(u))

    return u
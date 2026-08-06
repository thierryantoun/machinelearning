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
    ic_type = random.randint(subkey, (), 0, 5)  # 0: sinus, 1: gaussiennes, 2: polynomes, 3: constante, 4: rampe

    key, subkey = random.split(key)

    def make_sinus(subkey):
        a_k = random.uniform(subkey, (K,), minval=-1.0, maxval=1.0)
        key2, subkey2 = random.split(subkey)
        phase_k = random.uniform(subkey2, (K,), minval=0.0, maxval=2*jnp.pi)
        ks = jnp.arange(1, K+1)
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

    is_constante = (ic_type == 3)
    u = jax.lax.switch(ic_type, [make_sinus, make_gaussiennes, make_polynomes, make_constante, make_rampe], subkey)
    u = jnp.where(is_constante, u, u / jnp.max(jnp.abs(u)))

    return u
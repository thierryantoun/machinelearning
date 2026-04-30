import jax.numpy as jnp

N = 5000
n = 32
T = 0.5
K = 20
cfl = 0.5
a = 1.0
x = jnp.linspace(0, 1, n, endpoint=False)

batch_size = 64
nb_epoch = 200
n_batches = N // batch_size

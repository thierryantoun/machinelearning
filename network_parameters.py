import jax.numpy as jnp

N = 10000
n = 128
T = 0.5
K = 10
cfl = 0.5
x = jnp.linspace(0, 1, n, endpoint=False)

batch_size = 64
nb_epoch = 200
n_batches = N // batch_size

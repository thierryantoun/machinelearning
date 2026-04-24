import jax.numpy as jnp

N = 1000
n = 512
T = 0.5
K = 20
cfl = 0.5
x = jnp.linspace(0, 1, n, endpoint=False)

batch_size = 64
nb_epoch = 200
n_batches = N // batch_size

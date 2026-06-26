import jax.numpy as jnp

SOLVER = "burgers"

N = 5000
n = 128
T = 0.5
K = 16
cfl = 0.5
a = 1.0
x = jnp.linspace(0, 1, n, endpoint=False)
n_steps = 10

batch_size = 64
nb_epoch = 1000
n_batches = N // batch_size

if SOLVER == "advection":
    from advection_solver import advection_solver as solver
else:
    from burgers_solver import burgers_solver as solver

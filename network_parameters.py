import jax.numpy as jnp

SOLVER = "burgers"   # "advection" ou "burgers"

K          = 80
N_RANDOM       = 20000  # nombre de u0 i.i.d. tirés aléatoirement
N_TRAJ         = 10     # nombre de trajectoires longues
MULTIPLE_STEPS = 100    # nombre de paires (u_k, u_{k+1}) par trajectoire longue
N_TRAIN    = N_RANDOM + N_TRAJ * MULTIPLE_STEPS  # nombre total de paires de training
n          = 512
T          = 0.5
cfl        = 0.5
a          = 1.0      # vitesse pour l'advection
x          = jnp.linspace(0, 1, n, endpoint=False)
n_steps    = 50

batch_size = 128
nb_epoch   = 1000
n_batches  = N_TRAIN // batch_size

if SOLVER == "advection":
    from advection_solver import advection_solver as solver
else:
    from burgers_solver import burgers_solver as solver

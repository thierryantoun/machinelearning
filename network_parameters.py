import argparse

import jax.numpy as jnp

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", choices=["fno", "lgno"], default="fno",
                      help="Modèle à utiliser : fno ou lgno")
_args, _ = _parser.parse_known_args()

MODEL = _args.model

SOLVER = "burgers"   # "advection" ou "burgers"

K          = 40
N_TRAJ         = 10   # nombre de trajectoires longues
MULTIPLE_STEPS = 20   # nombre de paires (u_k, u_{n_steps+k}) par trajectoire longue
N_TRAIN    = N_TRAJ * MULTIPLE_STEPS  # nombre total de paires de training
n          = 128
T          = 1
cfl        = 0.5
a          = 1.0      # vitesse pour l'advection
x          = jnp.linspace(0, 1, n, endpoint=False)
n_steps    = 10

batch_size = 64
nb_epoch   = 100
n_batches  = N_TRAIN // batch_size
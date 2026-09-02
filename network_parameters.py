import argparse

import jax.numpy as jnp

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", choices=["fno", "lgno"], default="fno",
                      help="Modèle à utiliser : fno ou lgno")
_args, _ = _parser.parse_known_args()

MODEL = _args.model

SOLVER = "burgers"   # "advection" ou "burgers"

K          = 60
N_TRAJ         = 10   # nombre de trajectoires longues
MULTIPLE_STEPS = 20   # nombre de paires (u_k, u_{n_steps+k}) par trajectoire longue
N_TRAIN    = N_TRAJ * MULTIPLE_STEPS  # nombre total de paires de training
n          = 256
T          = 1
cfl        = 0.5
a          = 1.0      # vitesse pour l'advection
x          = jnp.linspace(0, 1, n, endpoint=False)
T_target   = 0.1
batch_size = 64
nb_epoch   = 100
n_batches  = N_TRAIN // batch_size

lambda_hf = 0.0

# ------------------------------------------------------------------
# À METTRE À True À CHAQUE CHANGEMENT DE STAGE (MULTIPLE_STEPS, lambda_hf, K, etc.)
# La loss n'est alors plus comparable au stage précédent : on repart d'un
# best_val vierge pour ne pas bloquer la sauvegarde / déclencher un early
# stopping prématuré sur une métrique qui n'a plus le même sens.
# Remettre à False si on reprend un stage déjà entamé sans rien changer.
# ------------------------------------------------------------------
NEW_STAGE = True
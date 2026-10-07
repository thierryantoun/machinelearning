import argparse

import jax.numpy as jnp

from stencil import stencil_size

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", choices=["fno", "lgno"], default="fno",
                      help="Modèle �|  utiliser : fno ou lgno")
_args, _ = _parser.parse_known_args()

MODEL = _args.model

SOLVER = "burgers"   # "advection" ou "burgers"

N_TRAJ         = 1   # nombre de trajectoires longues
MULTIPLE_STEPS = 2  # nombre de paires (u_k, u_{n_steps+k}) par trajectoire longue
N_TRAIN    = N_TRAJ * MULTIPLE_STEPS  # nombre total de paires de training
n          = 256
alpha      = 40 / 128            # fraction de modes de Fourier gardés par le FNO (fixe)
K          = int(alpha * n / 2)  # nb de modes (FNO et données initiales) : n=256 -> 40
T          = 1
cfl        = 0.5
a          = 1.0      # vitesse pour l'advection
x          = jnp.linspace(0, 1, n, endpoint=False)
T_target   = 0.1               # lambda = T_target * n = 25.6 à n=256

# Bornes des vitesses caractéristiques f'(u) = min u0, max u0 (cf. print de training.py)
# -> taille de la fenêtre (cône de dépendance, §3.4 du doc), fixe car l'entrée et la
# sortie du réseau en dépendent. lambda=25.6, u0 in [-1, 1] -> P=25, Q=26 (52 cellules)
A_MIN, A_MAX = -1.0, 1.0
P, Q       = stencil_size(A_MIN, A_MAX, T_target * n)
batch_size = min(64, N_TRAIN)   # borné pour les petits tests (N_TRAIN < 64)
nb_epoch   = 500
n_batches  = N_TRAIN // batch_size

lambda_hf   = 1.0
lambda_phys = 1.0   # poids de loss_phys ; mettre à 0 pour l'exclure entièrement de la loss

# ------------------------------------------------------------------
# Entraînement "model-in-the-loop" (correction de l'exposure bias) :
# en plus des paires (u0, u_final) issues du vrai solveur, on déroule
# le modèle courant (arrêt de gradient) depuis des IC fraîches puis on
# interroge le vrai solveur depuis l'état atteint pour ré-étiqueter
# correctement. Le réseau apprend ainsi �|  corriger ses propres erreurs
# accumulées, pas seulement �|  reproduire des trajectoires "propres".
# ------------------------------------------------------------------
ONPOLICY_ENABLED       = False
ONPOLICY_TRAJ          = N_TRAJ   # nb de trajectoires on-policy régénérées
ONPOLICY_MAX_STEPS     = 40       # profondeur max de rollout modèle avant relabelling
ONPOLICY_DEPTHS_PER_TRAJ = 4      # nb de profondeurs piochées par trajectoire (les 40 états
                                   # sont déj�|  calculés pour rien de plus, autant en garder
                                   # plusieurs -> ONPOLICY_TRAJ * ONPOLICY_DEPTHS_PER_TRAJ paires/epoch)
ONPOLICY_REGEN_EVERY   = 1        # régénère les paires on-policy tous les N epochs

# ------------------------------------------------------------------
# METTRE A True A CHAQUE CHANGEMENT DE STAGE (MULTIPLE_STEPS, lambda_hf, K, etc.)
# La loss n'est alors plus comparable au stage précédent : on repart d'un
# best_val vierge pour ne pas bloquer la sauvegarde / déclencher un early
# stopping prématuré sur une métrique qui n'a plus le même sens.
# Remettre �|  False si on reprend un stage déj�|  entamé sans rien changer.
# ------------------------------------------------------------------
NEW_STAGE = True

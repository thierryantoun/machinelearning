import argparse
import math

import jax.numpy as jnp

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", choices=["stencil_conv1D", "lgno"], default="stencil_conv1D",
                      help="Modèle �|  utiliser : stencil_conv1D ou lgno")
_args, _ = _parser.parse_known_args()

MODEL = _args.model

SOLVER = "euler"   

K          = 40   # nombre de fréquences des conditions initiales (initial_data.py), sans lien avec le réseau

N_TRAJ         = 10000  # nombre de trajectoires longues
MULTIPLE_STEPS = 200  # nombre de paires (u_k, u_{n_steps+k}) par trajectoire longue
N_TRAIN    = N_TRAJ * MULTIPLE_STEPS  # nombre total de paires de training
n          = 256
T          = 50     # temps physique final des rollouts à pas CFL adaptatif (test.py)
cfl        = 0.5
a          = 1.0      # vitesse pour l'advection
x          = jnp.linspace(0, 1, n, endpoint=False)
LAMBDA     = 25.6              # lambda = T * V / dx fixé, en cellules : l'onde la plus rapide parcourt LAMBDA cellules par bloc

# Taille de la fenêtre (cône de dépendance, §3.4 du doc), fixe car l'entrée et la sortie du
# réseau en dépendent. V n'est exact qu'au début du bloc : MARGE couvre la vitesse atteinte
# pendant le bloc (cf. print de training.py). lambda=25.6, MARGE=1 -> P=25, Q=26 (52 cellules)
MARGE      = 1.0
Q          = math.ceil(MARGE * LAMBDA)
P          = Q - 1
batch_size = min(64, N_TRAIN)   # borné pour les petits tests (N_TRAIN < 64)
nb_epoch   = 500
n_batches  = N_TRAIN // batch_size

lambda_hf   = 1.0
lambda_phys = 1.0   # poids de loss_phys ; mettre à 0 pour l'exclure entièrement de la loss
lambda_pos  = 0.1   # poids de loss_pos (positivité de rho et p prédits) ; 0 pour la désactiver
EPS_POS     = 0.5   # marge : pénalise les cellules où rho_pred < EPS_POS * rho_cible (idem p)

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
EPS_ADMISSIBLE         = 0.1      # état on-policy gardé si rho > EPS * min(rho_IC) et p > EPS * min(p_IC) partout

# ------------------------------------------------------------------
# METTRE A True A CHAQUE CHANGEMENT DE STAGE (MULTIPLE_STEPS, lambda_hf, K, etc.)
# La loss n'est alors plus comparable au stage précédent : on repart d'un
# best_val vierge pour ne pas bloquer la sauvegarde / déclencher un early
# stopping prématuré sur une métrique qui n'a plus le même sens.
# Remettre �|  False si on reprend un stage déj�|  entamé sans rien changer.
# ------------------------------------------------------------------
NEW_STAGE = True

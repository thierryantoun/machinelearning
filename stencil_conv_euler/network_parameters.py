import math

import jax.numpy as jnp

MODEL = "stencil_conv1D"

SOLVER = "euler"   

K          = 40   # nombre de fréquences des conditions initiales (initial_data.py), sans lien avec le réseau

N_TRAJ         = 150000  # nombre de trajectoires longues
MULTIPLE_STEPS = 1# nombre de paires (u_k, u_{n_steps+k}) par trajectoire longue
N_TRAIN    = N_TRAJ * MULTIPLE_STEPS  # nombre total de paires de training
n          = 256
T          = 0.1  # temps physique final des rollouts à pas CFL adaptatif (test.py)
cfl        = 0.5
a          = 1.0      # vitesse pour l'advection
x          = jnp.linspace(0, 1, n, endpoint=False)
LAMBDA     = 1             # lambda = T * V / dx fixé, en cellules : l'onde la plus rapide parcourt LAMBDA cellules par bloc

# Taille de la fenêtre (cône de dépendance, §3.4 du doc), fixe car l'entrée et la sortie du
# réseau en dépendent. V n'est exact qu'au début du bloc : MARGE, en cellules, s'ajoute à LAMBDA
# pour couvrir la distance réellement parcourue pendant le bloc (mesurée par mesure_Q.py : MARGE = Q conseillé - LAMBDA).
# Fenêtre de 2Q cellules : Q à gauche de la face (j-P..j) et Q à droite. LAMBDA=26, MARGE=6 -> P=31, Q=32 (64 cellules)
MARGE      = 10.0
Q          = math.ceil(LAMBDA + MARGE)
P          = Q - 1
batch_size = 128
nb_epoch   = 710   # plafond de sécurité par run (compté depuis l'epoch de reprise) : l'arrêt normal vient de MIN_LR (training.py)
n_batches  = N_TRAIN // batch_size

lambda_hf   = 1.0
PHYS_NORM   = "L1"  # norme de loss_phys : "L1" (sum|e| / sum|cible|, chocs moins étalés) ou "L2" (||e||_2 / ||cible||_2)
lambda_phys = 1.0   # poids de loss_phys ; mettre à 0 pour l'exclure entièrement de la loss
lambda_pos  = 1.0   # ABLATION (baseline 0.1 dans pkl/baseline_lambda_pos_0.1/) ; poids de loss_pos (positivité de rho et p prédits) ; 0 pour la désactiver
EPS_POS     = 0.5   # marge : pénalise les cellules où rho_pred < EPS_POS * rho_cible (idem p)
EPS_Q       = 1e-2  # plancher du dénominateur de la pénalité sur q : q_cible + EPS_Q * moyenne(q_cible) ; amplification max 1 / EPS_Q

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
EPS_ADMISSIBLE         = 1e-2     # état on-policy gardé si rho > EPS et p > EPS partout (seuil absolu : une double
                                   # détente exacte peut descendre sous le min de la CI, pas de principe du minimum)

# ------------------------------------------------------------------
# METTRE A True A CHAQUE CHANGEMENT DE STAGE (MULTIPLE_STEPS, lambda_hf, K, etc.)
# La loss n'est alors plus comparable au stage précédent : on repart d'un
# best_val vierge pour ne pas bloquer la sauvegarde / déclencher un early
# stopping prématuré sur une métrique qui n'a plus le même sens.
# Remettre �|  False si on reprend un stage déj�|  entamé sans rien changer.
# ------------------------------------------------------------------
NEW_STAGE = True

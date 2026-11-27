"""Mesure de Q (demi-largeur de la fenêtre du réseau, en cellules) à lancer AVANT training.py.

Pour chaque LAMBDA : bloc de durée T = LAMBDA dx / V0 (V0 = max(|u| + c) au début du bloc, comme T_bloc),
avancé par le vrai solveur, qui renvoie dist = nombre de cellules parcourues par l'onde la plus rapide pendant
le bloc. Q conseillé = ceil(max dist) sur les CI d'entraînement et de validation (mêmes clés que training.py),
puis P = Q - 1, fenêtre de 2Q cellules. Usage : python mesure_Q.py [N_CI] [LAMBDA1 LAMBDA2 ...]
"""
import math
import sys

import jax
import jax.numpy as jnp
from jax import random

from network_parameters import N_TRAJ, MULTIPLE_STEPS, LAMBDA, Q, x
from initial_data import generate_initial_data_euler, keys_non_constantes
from euler_solver import euler_solver, primitives, gamma

N_CI    = int(sys.argv[1]) if len(sys.argv) > 1 else N_TRAJ + max(1, N_TRAJ // 5)   # défaut : train + val
LAMBDAS = [float(l) for l in sys.argv[2:]] or [LAMBDA]
CHUNK   = 10000
n       = x.shape[0]


def V_max(U):
    rho, u, p = primitives(U)
    return jnp.max(jnp.abs(u) + jnp.sqrt(gamma * p / rho))


def dist_trajectoire(key, lam):
    """dist (cellules) de chacun des MULTIPLE_STEPS blocs de la trajectoire issue de la CI de clé key."""
    def bloc(U, _):
        U_next, _, _, dist = euler_solver(U, lam / n / V_max(U), return_dist=True)
        return U_next, dist
    return jax.lax.scan(bloc, generate_initial_data_euler(key), None, length=MULTIPLE_STEPS)[1]


_dist_chunk = jax.jit(jax.vmap(dist_trajectoire, in_axes=(0, None)))

# mêmes clés que training.py (CI d'entraînement puis de validation)
key_train = random.split(random.PRNGKey(0), 3)[1]
_, key_traj_train, key_traj_val = random.split(key_train, 3)
# ne génère que N_CI CI : d'abord N_tr de l'entraînement, puis N_va de la validation. Les tirages ne sont
# identiques à ceux de training.py qu'avec le défaut (N_CI = train + val) ; sinon, même distribution
N_VAL = max(1, N_TRAJ // 5)
N_tr  = min(N_CI, N_TRAJ)
N_va  = min(N_CI - N_tr, N_VAL)
keys  = keys_non_constantes(key_traj_train, N_tr)
if N_va > 0:
    keys = jnp.concatenate([keys, keys_non_constantes(key_traj_val, N_va)])

print(f"{keys.shape[0]} CI, {MULTIPLE_STEPS} bloc(s) par CI, n={n}, Q actuel (network_parameters) = {Q}")
print("vitesse CFL = max(|u| + s_ac) du solveur (majorant) ; physique = max(|u| + c) (cône physique)")
print(f"{'LAMBDA':>7} {'vitesse':>9} {'dist max':>9} {'99.9 %':>8} {'99 %':>8} {'médiane':>8} {'Q - LAMBDA':>11} {'Q conseillé':>12}")
for lam in LAMBDAS:
    D = jnp.concatenate([_dist_chunk(keys[i:i + CHUNK], lam) for i in range(0, keys.shape[0], CHUNK)]).reshape(-1, 2)
    for nom, d in (("CFL", D[:, 0]), ("physique", D[:, 1])):
        q999, q99, q50 = (float(v) for v in jnp.quantile(d, jnp.array([0.999, 0.99, 0.5])))
        Q_cons = math.ceil(float(jnp.max(d)))
        print(f"{lam:7g} {nom:>9} {float(jnp.max(d)):9.2f} {q999:8.2f} {q99:8.2f} {q50:8.2f} {Q_cons - lam:11g} {Q_cons:12d}")

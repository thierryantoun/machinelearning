"""Loss par type de condition initiale : justifie les proportions IC_PROBS de initial_data.py.

Pour chaque type, N_PAR_TYPE paires générées comme dans generate_trajectory (MULTIPLE_STEPS = 1) :
entrée normalisée par V = max|u|, cible = solveur de référence sur un bloc T_target.
loss_phys est ramenée à un dénominateur commun (moyenne de sum|cible| sur tous les types) :
sinon chaque type serait normalisé par sa propre amplitude et les valeurs ne seraient pas comparables.
loss_hf ne dépend d'aucun dénominateur.

Usage : python loss_par_type.py [params_1.pkl params_2.pkl ...]   (défaut : params_stencil_conv1D.pkl)
"""
import sys
import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from initial_data import generate_initial_data, IC_NAMES, IC_PROBS
from network_parameters import T_target, SOLVER
from loss import loss_fn, V_max

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver
else:
    from burgers_solver import burgers_solver as _solver

N_PAR_TYPE = 200
SEED       = 7
FICHIERS   = sys.argv[1:] or ["params_stencil_conv1D.pkl"]
COULEURS   = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def charger(f):
    with open(f, "rb") as fh:
        p = pickle.load(fh)
    if "kernel" in p["params"]:   # ancienne version (noyau déclaré à la main)
        q = dict(p["params"])
        q["lifting"] = {"kernel": q.pop("kernel"), "bias": q.pop("bias")}
        p = {"params": q}
    return p


# ---------- paires par type, même pipeline que generate_trajectory ----------
cible = jax.jit(jax.vmap(lambda u: _solver(u, T_target)[0]))
entrees, cibles = [], []
for t in range(len(IC_NAMES)):
    keys = random.split(random.PRNGKey(SEED + t), N_PAR_TYPE)
    u = jax.vmap(partial(generate_initial_data, ic_type=t))(keys)
    u = u / jax.vmap(V_max)(u)[:, None]
    entrees.append(u)
    cibles.append(cible(u))

den = float(jnp.mean(jnp.stack([jnp.mean(jnp.sum(jnp.abs(y), axis=1)) for y in cibles])))

# ---------- loss par type et par modèle ----------
res = {}
for f in FICHIERS:
    p = charger(f)
    phys, hf = [], []
    for u, y in zip(entrees, cibles):
        _, aux = loss_fn(p, u, y)
        den_type = float(jnp.mean(jnp.sum(jnp.abs(y), axis=1)))
        phys.append(float(aux["loss_phys"]) * den_type / den)   # dénominateur commun
        hf.append(float(aux["loss_hf"]))
    res[f] = (jnp.array(phys), jnp.array(hf))

probs_unif = jnp.full(len(IC_NAMES), 1.0 / len(IC_NAMES))
for f, (phys, hf) in res.items():
    tot = phys + hf
    part_unif  = probs_unif * tot / jnp.sum(probs_unif * tot)   # part de la loss totale portée par chaque type
    part_probs = IC_PROBS * tot / jnp.sum(IC_PROBS * tot)
    print(f"\n=== {f} ===")
    print(f"{'type':<15}{'phys':>9}{'hf':>9}{'total':>9}{'tirage unif.':>14}{'part loss':>11}"
          f"{'IC_PROBS':>10}{'part loss':>11}")
    for i in jnp.argsort(-tot):
        i = int(i)
        print(f"{IC_NAMES[i]:<15}{float(phys[i]):>9.5f}{float(hf[i]):>9.5f}{float(tot[i]):>9.5f}"
              f"{float(probs_unif[i]):>13.0%} {float(part_unif[i]):>10.1%}"
              f"{float(IC_PROBS[i]):>10.0%} {float(part_probs[i]):>10.1%}")
    print(f"loss moyenne : tirage uniforme {float(jnp.sum(probs_unif * tot)):.5f} | "
          f"IC_PROBS {float(jnp.sum(IC_PROBS * tot)):.5f}  (plus haute = dataset plus exigeant)")
    p_prop = tot / jnp.sum(tot)   # p_i ∝ L_i : plus de données là où l'erreur par paire est grande
    print("IC_PROBS proposé (p_i ∝ L_i, ordre IC_NAMES) : ["
          + ", ".join(f"{float(v):.3f}" for v in p_prop) + "]")

# ---------- figure : loss_phys, loss_hf, proportions ----------
ordre = list(map(int, jnp.argsort(-(res[FICHIERS[0]][0] + res[FICHIERS[0]][1]))))
noms  = [IC_NAMES[i] for i in ordre]
y     = jnp.arange(len(noms))
h     = 0.8 / max(len(FICHIERS), 2)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for k, f in enumerate(FICHIERS):
    phys, hf = res[f]
    off = (k - (len(FICHIERS) - 1) / 2) * h
    axs[0].barh(y + off, phys[jnp.array(ordre)], height=h * 0.9, color=COULEURS[k % 4], label=f)
    axs[1].barh(y + off, hf[jnp.array(ordre)], height=h * 0.9, color=COULEURS[k % 4], label=f)
axs[2].barh(y - 0.2, probs_unif[jnp.array(ordre)] * 100, height=0.36, color="#b0b0a8", label="tirage uniforme")
axs[2].barh(y + 0.2, IC_PROBS[jnp.array(ordre)] * 100, height=0.36, color="#1baf7a", label="IC_PROBS")
axs[0].set_title("loss_phys (dénominateur commun)")
axs[1].set_title("loss_hf")
axs[2].set_title("proportion du dataset (%)")
axs[0].set_yticks(y, noms)
axs[0].invert_yaxis()
for ax in axs:
    ax.grid(True, axis="x", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
axs[0].legend(fontsize=7, frameon=False, loc="lower right")
axs[2].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("loss_par_type.png", dpi=150)
print("\nFigure : loss_par_type.png")

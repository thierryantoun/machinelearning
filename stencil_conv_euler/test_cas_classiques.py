"""Cas tests classiques hors distribution : modèle (blocs) vs solveur de référence.

- Blast (type Woodward-Colella, version périodique de test_blast.py) : p = 1000 / 0.01 / 100,
  rapports de pression 1e5, très loin de l'entraînement (p dans [0.22, 4.5]).
- Brio-Wu, PARTIE HYDRODYNAMIQUE SEULEMENT : le vrai Brio-Wu est un tube à choc MHD
  (Bx = 0.75, By = +-1, gamma = 2), impossible avec Euler. On garde rho = 1 / 0.125,
  p = 1 / 0.1, u = 0, avec le gamma du solveur (5/3) : c'est un tube de Sod.
- Sod : rho = 1 / 0.125, p = 1 / 0.1, u = 0, saut en x = 0.5, t = 0.2 (gamma du solveur, 5/3).
Domaine périodique [0, 1) : un second saut existe au raccord x = 0.
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_parameters import MODEL, SOLVER
from loss import rollout, V_max, T_bloc
from euler_solver import euler_solver, primitives, gamma

with open(f"params_{MODEL}_{SOLVER}.pkl", "rb") as f:
    params = pickle.load(f)

RESOLUTIONS = [256, 1024]


def euler_state(rho, u, p):
    return jnp.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2])


def blast(x):
    p = jnp.where(x < 0.1, 1000.0, jnp.where(x > 0.9, 100.0, 0.01))
    return euler_state(jnp.ones_like(x), jnp.zeros_like(x), p)


def brio_wu_hydro(x):
    return euler_state(jnp.where(x < 0.5, 1.0, 0.125), jnp.zeros_like(x), jnp.where(x < 0.5, 1.0, 0.1))


def sod(x):
    return euler_state(jnp.where(x < 0.5, 1.0, 0.125), jnp.zeros_like(x), jnp.where(x < 0.5, 1.0, 0.1))


CASES = [
    # (nom, CI, temps final, échelle log pour p)
    ("blast", blast, 0.012, True),
    ("brio_wu_hydro", brio_wu_hydro, 0.2, False),
    ("sod", sod, 0.2, False),
]

TITRES = {
    "blast": "Blast (Woodward-Colella périodique)",
    "brio_wu_hydro": "Brio-Wu, partie hydrodynamique seule (sans B, γ = 5/3) — équivaut à un tube de Sod",
    "sod": "Tube de Sod (γ = 5/3, domaine périodique)",
}

C_REF, C_MOD = "#2a78d6", "#eb6834"

for name, ic, t_final, log_p in CASES:
    fig, axes = plt.subplots(3, len(RESOLUTIONS), figsize=(6 * len(RESOLUTIONS), 9), sharex=True, squeeze=False)
    for col, N in enumerate(RESOLUTIONS):
        x = (jnp.arange(N) + 0.5) / N
        u0 = ic(x)
        u_ref = euler_solver(u0, t_final)[0]
        u_mod, n_blocks = rollout(params, u0, t_final)
        fini = bool(jnp.all(jnp.isfinite(u_mod)))
        prim0, prim_ref, prim_mod = primitives(u0), primitives(u_ref), primitives(u_mod)
        rel = [float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b)) for a, b in zip(prim_mod, prim_ref)]
        print(f"[{name}] N={N} t={t_final:g} : {int(n_blocks)} blocs modèle (T_bloc initial {float(T_bloc(u0)):.2e}, "
              f"V0={float(V_max(u0)):.2f}), fini={fini}, min p modèle={float(jnp.nanmin(prim_mod[2])):.3e}, "
              f"err. rel. L2 rho={rel[0]:.2e} u={rel[1]:.2e} p={rel[2]:.2e}")
        for row, (lab, v0, vr, vm) in enumerate(zip(("ρ", "u", "p"), prim0, prim_ref, prim_mod)):
            ax = axes[row][col]
            ax.plot(x, v0, color="gray", ls="--", lw=0.8, alpha=0.6, label="t = 0")
            ax.plot(x, vr, color=C_REF, lw=1.6, label="solveur")
            ax.plot(x, vm, color=C_MOD, lw=1.4, ls=":", label="modèle")
            ax.set_ylabel(lab)
            ax.grid(True, color="#e6e5e1", lw=0.6)
            if row == 2 and log_p:
                ax.set_yscale("log")
        axes[0][col].set_title(f"N={N}, t={t_final:g}, {int(n_blocks)} blocs" + ("" if fini else "  (NaN !)"))
        axes[2][col].set_xlabel("x")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(TITRES[name])
    fig.tight_layout()
    out = f"cas_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure -> {out}")

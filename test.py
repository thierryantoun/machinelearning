import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from network_parameters import x, SOLVER, T_target
from loss import predict_F

if SOLVER == "advection":
    from advection_solver import advection_solver as _active_solver
else:
    from burgers_solver import burgers_solver as _active_solver

# Le solveur prend T_target (temps physique fixe) au lieu de n_steps.
solver = lambda u: _active_solver(u, T_target)

dx = x[1] - x[0]

with open("params_fno_autocorrection_K40_lambda05_kappa05.pkl", "rb") as f:
    params = pickle.load(f)


def step(u, t_block):
    F = predict_F(params, u)
    u_next = u - t_block / dx * (F - jnp.roll(F, 1, axis=0))
    return u_next, None


@partial(jax.jit, static_argnames=("n_steps",))
def solver_rollout(u0, n_steps):
    def solver_block(u, _):
        u_next, _, _ = solver(u)
        return u_next, None
    u_target, _ = jax.lax.scan(solver_block, u0, None, length=n_steps)
    return u_target


@partial(jax.jit, static_argnames=("n_steps", "correction_every"))
def model_rollout(u0, n_steps, correction_every=None):
    """Rollout du modèle. Si correction_every est donné, tous les
    `correction_every` blocs la solution est réinjectée un pas dans le vrai
    schéma numérique (burgers_solver) à la place de la prédiction du modèle.

    Décoré @jax.jit (n_steps/correction_every statiques) : sans ça, chaque
    appel Python retrace et recompile le lax.scan depuis zéro, même pour un
    n_steps déjà vu (le cache de compilation de JAX vit sur l'objet jit, pas
    sur les closures internes). Avec le jit, la compilation est faite une
    fois par (n_steps, correction_every) et réutilisée pour les 3 fonctions
    test et les rollouts suivants avec le même n_steps."""
    def model_block(u, i):
        u_next, _ = step(u, T_target)
        if correction_every:
            do_correct = ((i + 1) % correction_every) == 0
            u_next = jax.lax.cond(do_correct, lambda uu: solver(uu)[0], lambda uu: uu, u_next)
        return u_next, None
    u_pred, _ = jax.lax.scan(model_block, u0, jnp.arange(n_steps))
    return u_pred


CORRECTION_EVERY = 100


def rollout(u0, n_steps):
    t0 = time.perf_counter()
    u_target = solver_rollout(u0, n_steps)
    jax.block_until_ready(u_target)
    t_solver = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_pred = model_rollout(u0, n_steps)
    jax.block_until_ready(u_pred)
    t_model = time.perf_counter() - t0

    mse = float(jnp.mean((u_pred - u_target) ** 2))

    # Inutile de calculer/afficher la version "corrigée" si CORRECTION_EVERY
    # est désactivé (<=0) ou si la correction ne se déclenche jamais sur ce
    # nombre de blocs (n_steps < CORRECTION_EVERY).
    has_correction = CORRECTION_EVERY > 0 and n_steps >= CORRECTION_EVERY
    if has_correction:
        t0 = time.perf_counter()
        u_pred_corr = model_rollout(u0, n_steps, correction_every=CORRECTION_EVERY)
        jax.block_until_ready(u_pred_corr)
        t_model_corr = time.perf_counter() - t0
        mse_corr = float(jnp.mean((u_pred_corr - u_target) ** 2))
    else:
        u_pred_corr = None
        mse_corr = None
        t_model_corr = None

    return u_target, u_pred, u_pred_corr, mse, mse_corr, t_solver, t_model, t_model_corr


# Fonctions initiales absentes du dataset d'entraînement (initial_data.py ne
# génère que : sinus multi-fréquences, somme de gaussiennes, polynôme, constante,
# rampe tanh).
u0_triangle = 2 / jnp.pi * jnp.arcsin(jnp.sin(2 * jnp.pi * x))
u0_carre    = jnp.sign(jnp.sin(2 * jnp.pi * x))
u0_paquet   = jnp.exp(-100 * (x - 0.5) ** 2) * jnp.sin(8 * jnp.pi * x)

test_functions = {
    "triangle":     u0_triangle,
    "carre":        u0_carre,
    "paquet_onde":  u0_paquet,
}

multiple_steps_list = [1, 10, 50, 100, 2000]

n_cols = 3
n_rows = -(-len(multiple_steps_list) // n_cols)  # ceil

for name, u0 in test_functions.items():
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True)
    axes_flat = axes.flat
    for ax, n_steps in zip(axes_flat, multiple_steps_list):
        u_target, u_pred, u_pred_corr, mse, mse_corr, t_solver, t_model, t_model_corr = rollout(u0, n_steps)
        ax.plot(x, u0,       label='u₀',           linestyle='--', alpha=0.5)
        ax.plot(x, u_target, label='cible',         linewidth=1.5)
        ax.plot(x, u_pred,   label='prédit (pur)',  linewidth=1.5, linestyle=':')

        speedup = t_solver / t_model if t_model > 0 else float('nan')
        temps_str = f"solveur={t_solver*1e3:.1f}ms  modèle={t_model*1e3:.1f}ms  (×{speedup:.1f})"

        if u_pred_corr is not None:
            surcout = t_model_corr / t_model if t_model > 0 else float('nan')
            temps_str += f"  modèle corrigé={t_model_corr*1e3:.1f}ms  (×{surcout:.1f} vs pur)"
            ax.plot(x, u_pred_corr, label=f'prédit (corrigé/{CORRECTION_EVERY})', linewidth=1.5, linestyle='-.')
            ax.set_title(f"{n_steps} blocs (MSE pur={mse:.2e}, corrigé={mse_corr:.2e})\n{temps_str}", fontsize=10)
            print(f"[{name}] multiple_steps={n_steps:4d}  MSE pur={mse:.6f}  MSE corrigé={mse_corr:.6f}  "
                  f"solveur={t_solver:.4f}s  modèle={t_model:.4f}s  modèle corrigé={t_model_corr:.4f}s  (×{surcout:.1f})")
        else:
            ax.set_title(f"{n_steps} blocs (MSE={mse:.2e})\n{temps_str}", fontsize=10)
            print(f"[{name}] multiple_steps={n_steps:4d}  MSE={mse:.6f}  "
                  f"solveur={t_solver:.4f}s  modèle={t_model:.4f}s  (×{speedup:.1f})")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(multiple_steps_list):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    for ax in axes.flat[max(0, len(multiple_steps_list) - n_cols):len(multiple_steps_list)]:
        ax.set_xlabel('x')
    fig.suptitle(f"Rollout — fonction test « {name} » (T_target={T_target})")
    fig.tight_layout()
    fig.savefig(f"test_{name}.png", dpi=150)
    print(f"Figure sauvegardée : test_{name}.png")

plt.show()

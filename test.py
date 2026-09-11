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

with open("params_fno_no_correction_new_ic_T01_lambdahf005.pkl", "rb") as f:
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


CORRECTION_EVERY = 8


@partial(jax.jit, static_argnames=("n_steps", "correction_every"))
def error_growth(u0, n_steps, correction_every=None):
    """Croissance de l'erreur bloc par bloc.

    On avance en parallèle trois trajectoires depuis le même u0 :
      - u_true  : le vrai schéma numérique (référence) ;
      - u_model : le modèle pur ;
      - u_corr  : le modèle avec réinjection un pas dans le vrai schéma
                  numérique tous les `correction_every` blocs (à la place de
                  la prédiction), comme model_rollout(..., correction_every).
    À chaque bloc n on mesure la norme L2 physique
    ‖v‖ = sqrt(dx · Σ v²) de (u_model − u_true), (u_corr − u_true) et de
    u_true lui-même (pour situer l'erreur par rapport au signal).

    Renvoie (errs_pur, errs_corr, norms_true), chacun de longueur n_steps ;
    si correction_every est None, errs_corr est identique à errs_pur.
    L'abscisse associée est t_n = n · T_target.
    """
    def block(carry, i):
        u_model, u_corr, u_true = carry
        u_true_next = solver(u_true)[0]

        u_model_next, _ = step(u_model, T_target)

        u_corr_next, _ = step(u_corr, T_target)
        if correction_every:
            do_correct = ((i + 1) % correction_every) == 0
            u_corr_next = jax.lax.cond(
                do_correct, lambda uu: solver(uu)[0], lambda uu: uu, u_corr_next
            )

        err_pur = jnp.sqrt(dx * jnp.sum((u_model_next - u_true_next) ** 2))
        err_corr = jnp.sqrt(dx * jnp.sum((u_corr_next - u_true_next) ** 2))
        norm_true = jnp.sqrt(dx * jnp.sum(u_true_next ** 2))
        return (u_model_next, u_corr_next, u_true_next), (err_pur, err_corr, norm_true)

    _, (errs_pur, errs_corr, norms_true) = jax.lax.scan(
        block, (u0, u0, u0), jnp.arange(n_steps)
    )
    return errs_pur, errs_corr, norms_true


# None -> aligne la courbe d'erreur sur le temps physique max des figures de
# prédiction (max(PHYSICAL_TIMES)) ; sinon on force un temps physique explicite.
ERROR_GROWTH_TIME = None


def _bench(fn, *args, warmup=2, repeat=5):
    """Chronométrage propre : on lance d'abord `warmup` appels non mesurés
    (block_until_ready inclus) pour absorber la compilation XLA et le premier
    passage "à froid" du GPU, puis on mesure `repeat` appels et on renvoie le
    minimum (le moins bruité — jitter OS, autres kernels, etc.). Sans ça, le
    tout premier appel d'un (n_steps, correction_every) donné mélange compile
    + exécution : c'est ce qui faisait paraître la 1ʳᵉ fonction test (triangle)
    3× plus lente que les suivantes et rendait la comparaison
    correction_every=2 vs 8 ininterprétable."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return out, best


def rollout(u0, n_steps):
    u_target, t_solver = _bench(solver_rollout, u0, n_steps)
    u_pred, t_model = _bench(model_rollout, u0, n_steps)

    mse = float(jnp.mean((u_pred - u_target) ** 2))

    # Inutile de calculer/afficher la version "corrigée" si CORRECTION_EVERY
    # est désactivé (<=0) ou si la correction ne se déclenche jamais sur ce
    # nombre de blocs (n_steps < CORRECTION_EVERY).
    has_correction = CORRECTION_EVERY > 0 and n_steps >= CORRECTION_EVERY
    if has_correction:
        model_rollout_corr = partial(model_rollout, correction_every=CORRECTION_EVERY)
        u_pred_corr, t_model_corr = _bench(model_rollout_corr, u0, n_steps)
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

# Temps physiques fixes (indépendants de T_target) sur lesquels comparer les
# rollouts : n_steps = round(t / T_target). Comme ça, changer T_target dans
# network_parameters.py ne fausse plus la comparaison vitesse/précision
# (avant, "2000 blocs" représentait un temps physique différent selon
# T_target, ce qui rendait les runs T_target=0.1 vs 0.2 non comparables).
PHYSICAL_TIMES = [0.1, 1, 5, 10, 200]
multiple_steps_list = [max(1, round(t / T_target)) for t in PHYSICAL_TIMES]

n_cols = 3
n_rows = -(-len(multiple_steps_list) // n_cols)  # ceil

for name, u0 in test_functions.items():
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True)
    axes_flat = axes.flat
    for ax, n_steps in zip(axes_flat, multiple_steps_list):
        t_phys = n_steps * T_target
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
            ax.set_title(f"t={t_phys:g} ({n_steps} blocs)  MSE pur={mse:.2e}, corrigé={mse_corr:.2e}\n{temps_str}", fontsize=10)
            print(f"[{name}] t={t_phys:g} (n_steps={n_steps:4d})  MSE pur={mse:.6f}  MSE corrigé={mse_corr:.6f}  "
                  f"solveur={t_solver:.4f}s  modèle={t_model:.4f}s  modèle corrigé={t_model_corr:.4f}s  (×{surcout:.1f})")
        else:
            ax.set_title(f"t={t_phys:g} ({n_steps} blocs)  MSE={mse:.2e}\n{temps_str}", fontsize=10)
            print(f"[{name}] t={t_phys:g} (n_steps={n_steps:4d})  MSE={mse:.6f}  "
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

# ------------------------------------------------------------------
# Courbe de croissance de l'erreur : ‖ε_n‖_L2 = ‖û_n(modèle) − u_n(vrai)‖
# en fonction de t_n = n · T_target, pour chaque fonction test.
# ------------------------------------------------------------------
error_growth_time = ERROR_GROWTH_TIME if ERROR_GROWTH_TIME is not None else max(PHYSICAL_TIMES)
n_steps_err = max(1, round(error_growth_time / T_target))
t_n = jnp.arange(1, n_steps_err + 1) * T_target
has_corr = CORRECTION_EVERY > 0 and n_steps_err >= CORRECTION_EVERY

fig_err, ax_err = plt.subplots(figsize=(8, 5))
for name, u0 in test_functions.items():
    errs_pur, errs_corr, norms_true = error_growth(
        u0, n_steps_err, correction_every=CORRECTION_EVERY if has_corr else None
    )
    jax.block_until_ready((errs_pur, errs_corr, norms_true))
    line, = ax_err.plot(t_n, errs_pur, label=f"{name} — pur", linewidth=1.5)
    if has_corr:
        ax_err.plot(t_n, errs_corr, color=line.get_color(), linestyle="--",
                    linewidth=1.5, label=f"{name} — corrigé/{CORRECTION_EVERY}")
        print(f"[{name}] ‖ε‖_L2 pur : t={T_target:g} → {float(errs_pur[0]):.3e}   "
              f"t={float(t_n[-1]):g} → {float(errs_pur[-1]):.3e}    "
              f"corrigé : t={float(t_n[-1]):g} → {float(errs_corr[-1]):.3e}")
    else:
        print(f"[{name}] ‖ε‖_L2 : t={T_target:g} → {float(errs_pur[0]):.3e}   "
              f"t={float(t_n[-1]):g} → {float(errs_pur[-1]):.3e}")

ax_err.set_xlabel("t_n = n · T_target")
ax_err.set_ylabel("‖ε_n‖_L2  =  ‖û_n − u_n(vrai)‖")
ax_err.set_yscale("log")
ax_err.set_title(
    f"Croissance de l'erreur du modèle (T_target={T_target}"
    + (f", correction tous les {CORRECTION_EVERY} blocs)" if has_corr else ")")
)
ax_err.grid(True, alpha=0.3, which="both")
ax_err.legend()
fig_err.tight_layout()
fig_err.savefig("error_growth.png", dpi=150)
print("Figure sauvegardée : error_growth.png")

plt.show()

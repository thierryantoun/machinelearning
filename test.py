import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from network_parameters import x, SOLVER, T_target, cfl
from loss import predict_F

if SOLVER == "advection":
    from advection_solver import advection_solver as _active_solver
else:
    from burgers_solver import burgers_solver as _active_solver

# Le solveur prend T_target (temps physique fixe) au lieu de n_steps.
solver = lambda u: _active_solver(u, T_target)

dx = x[1] - x[0]

with open("params_fno.pkl", "rb") as f:
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


CORRECTION_EVERY = 0


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


# Fonctions initiales absentes du dataset d'entraînement (initial_data.py ne
# génère que : sinus multi-fréquences, somme de gaussiennes, polynôme, constante,
# rampe tanh).
u0_triangle = 2 / jnp.pi * jnp.arcsin(jnp.sin(2 * jnp.pi * x))
u0_carre    = jnp.sign(jnp.sin(2 * jnp.pi * x))
u0_paquet   = jnp.exp(-100 * (x - 0.5) ** 2) * jnp.sin(8 * jnp.pi * x)
u0_sinus_simple = jnp.sin(2 * jnp.pi * x)
u0_somme_sinus  = (
    1.0 * jnp.sin(2 * jnp.pi * 1 * x)
    + 0.5 * jnp.sin(2 * jnp.pi * 3 * x + 0.7)
    + 0.3 * jnp.sin(2 * jnp.pi * 7 * x + 2.1)
)
u0_somme_sinus = u0_somme_sinus / jnp.max(jnp.abs(u0_somme_sinus))
u0_riemann = jnp.where(x < 0.5, 1.0, -0.5)  # probleme de Riemann simple (une discontinuite, cas test standard pour la formation de choc en Burgers)

test_functions = {
    "triangle":     u0_triangle,
    "carre":        u0_carre,
    "paquet_onde":  u0_paquet,
    "sinus_simple": u0_sinus_simple,
    "somme_sinus":  u0_somme_sinus,
    "riemann":      u0_riemann,
}

# Temps physiques fixes (indépendants de T_target) sur lesquels comparer les
# rollouts : n_steps = round(t / T_target). Comme ça, changer T_target dans
# network_parameters.py ne fausse plus la comparaison vitesse/précision
# (avant, "2000 blocs" représentait un temps physique différent selon
# T_target, ce qui rendait les runs T_target=0.1 vs 0.2 non comparables).
PHYSICAL_TIMES = [0.1, 1, 5, 10, 200]
multiple_steps_list = [max(1, round(t / T_target)) for t in PHYSICAL_TIMES]

# ------------------------------------------------------------------
# Snapshots : comparaison cible vs modèle (pur et corrigé) à un ou
# plusieurs instants fixes du rollout, plutôt qu'un film complet.
# Modifier SNAPSHOT_TIMES pour choisir les instants à tracer.
# ------------------------------------------------------------------
SNAPSHOT_TIMES = [0.5]  # temps physiques auxquels tracer un plot (ex: [1, 10, 50, 200])
SNAPSHOT_FUNCTIONS = ["triangle", "carre", "somme_sinus", "sinus_simple", "riemann"]


def make_snapshot(name, u0, t_phys):
    n_steps = max(1, round(t_phys / T_target))
    has_corr = CORRECTION_EVERY > 0 and n_steps >= CORRECTION_EVERY

    u_true = solver_rollout(u0, n_steps)
    u_pred = model_rollout(u0, n_steps)
    u_corr = model_rollout(u0, n_steps, correction_every=CORRECTION_EVERY) if has_corr else None
    jax.block_until_ready((u_true, u_pred, u_corr))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, u0, label='u₀', linestyle='--', alpha=0.4, color='gray')
    ax.plot(x, u_true, label='cible', linewidth=1.5)
    ax.plot(x, u_pred, label='prédit (pur)', linewidth=1.5, linestyle=':')
    if has_corr:
        ax.plot(x, u_corr, label=f'prédit (corrigé/{CORRECTION_EVERY})', linewidth=1.5, linestyle='-.')
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(f"« {name} »  —  t={t_phys:g}  (T_target={T_target})")
    fig.tight_layout()
    out_path = f"snapshot_{name}_t{t_phys:g}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Snapshot sauvegardé : {out_path}")


for name in SNAPSHOT_FUNCTIONS:
    for t_phys in SNAPSHOT_TIMES:
        make_snapshot(name, test_functions[name], t_phys)

if SOLVER != "advection":
    from burgers_solver import flux as _burgers_flux

    def burgers_step_cfl(u):
        "Un pas Godunov, dt = cfl*dx/max|u|, jamais clampé sur T_target."
        dt = cfl * dx / (jnp.max(jnp.abs(u)) + 1e-10)
        f_face = _burgers_flux(u, jnp.roll(u, -1))
        u_new = u - dt / dx * (f_face - jnp.roll(f_face, 1))
        return u_new, dt

    @jax.jit
    def burgers_rollout_cfl(u0, t_final):
        "Solveur pas-à-pas classique : avance en pas dt naturels jusqu'à t_final."
        def cond_fn(carry):
            u, t, n = carry
            return t < t_final

        def body_fn(carry):
            u, t, n = carry
            u_new, dt = burgers_step_cfl(u)
            return (u_new, t + dt, n + 1)

        return jax.lax.while_loop(cond_fn, body_fn, (u0, 0.0, 0))

    print("\n--- Perf : modèle (blocs T_target) vs solveur pas-à-pas ---")
    for name, u0 in test_functions.items():
        for t_phys in PHYSICAL_TIMES:
            n_steps_model = max(1, round(t_phys / T_target))

            u_model, t_model = _bench(model_rollout, u0, n_steps_model)
            (u_solver, _, n_dt_steps), t_solver = _bench(burgers_rollout_cfl, u0, t_phys)

            mse = float(jnp.mean((u_model - u_solver) ** 2))
            mse_norm = mse / (float(jnp.mean(u_solver ** 2)) + 1e-12)
            speedup = t_solver / t_model if t_model > 0 else float('nan')
            print(f"[{name}] t={t_phys:g}  modèle={n_steps_model} blocs T_target ({t_model*1e3:.2f}ms)  "
                  f"solveur={int(n_dt_steps)} pas dt ({t_solver*1e3:.2f}ms)  MSE={mse:.6f}  MSE_norm={mse_norm:.6f}  (×{speedup:.1f})")

# ------------------------------------------------------------------
# Erreur d'énergie par bande de fréquence, en fonction de l'horizon de
# rollout (n_steps croissant), avec et sans correction.
#   E_B(t)   = Σ_{k∈bande} |û(k,t)|²             (énergie spectrale de la bande)
#   ε_B(t)   = |E_B,pred(t) − E_B,true(t)| / (E_B,true(t) + ε)
# Bandes identiques à la table de référence (grille n=256 -> rfft sur 129
# bins, k=0..128) : low k∈[1,5], mid k∈[6,15], hi1 k∈[16,31], hi2 k∈[32,47],
# hi3 k∈[48,63], hi4 k∈[64,95], hi5 k∈[96,128].
# ------------------------------------------------------------------
FREQ_BANDS = {
    "low": (1, 5), "mid": (6, 15), "hi1": (16, 31), "hi2": (32, 47),
    "hi3": (48, 63), "hi4": (64, 95), "hi5": (96, 128),
}
EPS_BAND = 1e-12


def band_energy(u):
    "E_B = Σ_k |û(k)|² pour k dans chaque bande de FREQ_BANDS."
    power = jnp.abs(jnp.fft.rfft(u, axis=-1)) ** 2
    return {name: jnp.sum(power[..., k0:k1 + 1], axis=-1) for name, (k0, k1) in FREQ_BANDS.items()}


def band_error(u_pred, u_true):
    "ε_B par bande, entre un champ prédit et sa référence."
    e_pred, e_true = band_energy(u_pred), band_energy(u_true)
    return {name: jnp.abs(e_pred[name] - e_true[name]) / (e_true[name] + EPS_BAND) for name in FREQ_BANDS}


def print_band_table(title, correction_every):
    print(f"\n--- {title} ---")
    print(f"{'traj':<12}{'n_steps':>8}" + "".join(f"{b:>12}" for b in FREQ_BANDS))
    for name, u0 in test_functions.items():
        for n_steps in multiple_steps_list:
            u_true = solver_rollout(u0, n_steps)
            u_pred = model_rollout(u0, n_steps, correction_every=correction_every)
            errs = band_error(u_pred, u_true)
            row = f"{name:<12}{n_steps:>8}" + "".join(f"{float(errs[b]):>12.2e}" for b in FREQ_BANDS)
            print(row)


print_band_table("Erreur par bande de fréquence — SANS correction", correction_every=None)
print_band_table(f"Erreur par bande de fréquence — AVEC correction (every={CORRECTION_EVERY})",
                  correction_every=CORRECTION_EVERY)

# ------------------------------------------------------------------
# Courbe de croissance de l'erreur : ‖ε_n‖_L2 = ‖û_n(modèle) − u_n(vrai)‖
# en fonction de t_n = n · T_target, pour chaque fonction test.
# ------------------------------------------------------------------
error_growth_time = ERROR_GROWTH_TIME if ERROR_GROWTH_TIME is not None else max(PHYSICAL_TIMES)
n_steps_err = max(1, round(error_growth_time / T_target))
t_n = jnp.arange(1, n_steps_err + 1) * T_target

fig_err, ax_err = plt.subplots(figsize=(8, 5))
for name, u0 in test_functions.items():
    errs_pur, _, norms_true = error_growth(u0, n_steps_err)
    jax.block_until_ready((errs_pur, norms_true))
    ax_err.plot(t_n, errs_pur, linewidth=1.5, label=name)
    print(f"[{name}] ‖ε‖_L2 : t={T_target:g} → {float(errs_pur[0]):.3e}   "
          f"t={float(t_n[-1]):g} → {float(errs_pur[-1]):.3e}")

ax_err.set_xlabel("t_n = n · T_target")
ax_err.set_ylabel("‖ε_n‖_L2  =  ‖û_n − u_n(vrai)‖")
ax_err.set_yscale("log")
ax_err.set_title(f"Croissance de l'erreur du modèle (T_target={T_target})")
ax_err.grid(True, alpha=0.3, which="both")
ax_err.legend()
fig_err.tight_layout()
fig_err.savefig("error_growth.png", dpi=150)
print("Figure sauvegardée : error_growth.png")

plt.show()

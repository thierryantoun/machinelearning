import math
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from network_parameters import SOLVER, LAMBDA, n, cfl, T
from loss import predict_F, V_max

if SOLVER == "advection":
    from advection_solver import advection_solver as _active_solver
else:
    from burgers_solver import burgers_solver as _active_solver

with open("params_stencil_conv1D.pkl", "rb") as f:
    params = pickle.load(f)

# Params entraînés avec l'ancienne version (noyau déclaré à la main : params/kernel, params/bias)
# -> rangés sous params/lifting pour le nn.Conv actuel. Même calcul, écart 0.
if "kernel" in params["params"]:
    p = dict(params["params"])
    p["lifting"] = {"kernel": p.pop("kernel"), "bias": p.pop("bias")}
    params = {"params": p}

# ------------------------------------------------------------------
# Évaluation multi-résolution à lambda = T / dx fixé (doc §2, protocole
# étape 2) : le modèle est entraîné à n = N0, puis évalué SANS
# réentraînement sur d'autres grilles avec T_N = LAMBDA / N.
# Domaine périodique [0, 1) -> dx = 1 / N. Les mêmes P, Q (fenêtre en
# cellules) servent sur toutes les grilles : le noyau de la conv (P+Q+1
# poids) est le même quelle que soit N.
#
# Tout ce qui dépend de la grille (dx, T, nombre de pas) est déduit de la
# taille de u : les fonctions jit ci-dessous se recompilent une fois par
# résolution, rien d'autre à changer.
# ------------------------------------------------------------------
RESOLUTIONS = [256,512,1024]   # grille d'entraînement seule pour l'instant ; [256, 512, 1024] pour le multi-résolution
PERF_RESOLUTIONS = [256, 512, 1024]   # perf modèle vs solveur : grille d'entraînement + résolutions plus fines (λ fixe)


def make_grid(N):
    return jnp.linspace(0, 1, N, endpoint=False)


def T_of(N):
    "Pas de temps d'un bloc sur la grille N, à lambda fixé."
    return LAMBDA / N


def n_steps_of(t_phys, N):
    return max(1, round(t_phys / T_of(N)))


def n_blocks_max(t_phys, N):
    "Majorant du nombre de blocs adaptatifs pour atteindre t_phys (blocs >= T_N tant que max|u| <= 1)."
    return max(1, math.ceil(t_phys / T_of(N) - 1e-9))


def T_bloc(u):
    "Durée d'un bloc adaptatif : T_N / V(u), V = max|u| (cf. loss.bloc)."
    return T_of(u.shape[-1]) / V_max(u)


def solver(u):
    "Un bloc du vrai schéma, sur la même durée T_bloc(u) qu'un bloc du modèle."
    return _active_solver(u, T_bloc(u))


def _step_raw(u):
    "Pas brut du réseau à lambda fixé, sans normalisation : valable pour max|u| <= 1."
    F = predict_F(params, u)
    return u - LAMBDA * (F - jnp.roll(F, 1, axis=-1))


def step(u):
    "Un bloc adaptatif du modèle : normalise par V, pas brut, remise à l'échelle ; dure T_bloc(u)."
    V = V_max(u)
    return V * _step_raw(u / V)


def l2(v):
    "Norme L2 physique sqrt(dx * sum v^2), dx = 1/N : comparable d'une grille à l'autre."
    return jnp.sqrt(jnp.sum(v ** 2, axis=-1) / v.shape[-1])


@partial(jax.jit, static_argnames=("n_steps",))
def solver_rollout(u0, n_steps):
    def solver_block(u, _):   # référence à blocs fixes T_N : arrive au temps physique n_steps * T_N
        return _active_solver(u, T_of(u.shape[-1]))[0], None
    u_target, _ = jax.lax.scan(solver_block, u0, None, length=n_steps)
    return u_target


@jax.jit
def model_rollout(u0, t_final=T):
    """Rollout à pas CFL adaptatif jusqu'au temps physique t_final.

    Invariance d'échelle de Burgers : v(x, t) = alpha u(x, alpha t). Pour tout
    alpha >= max|u|, u / alpha a une vitesse max <= 1 (domaine d'entraînement) ;
    un bloc du modèle sur u / alpha (lambda inchangé) fait avancer u de T_N / alpha.
      - alpha = max|u|                  -> pas T_cfl = T_N / max|u| (cône = stencil)
      - alpha = T_N / (t_final - t)     -> dernier pas raccourci pour tomber pile sur t_final
    On prend le plus grand des deux : le pas ne dépasse jamais T_cfl ni t_final,
    et max|u| -> 0 ne pose pas de problème (le pas est borné par le temps restant).

    Renvoie (u(t_final), nombre de blocs).
    """
    T_N = T_of(u0.shape[-1])

    def cond_fn(carry):
        _, t, _ = carry
        return t < t_final

    def body_fn(carry):
        u, t, n_blocks = carry
        alpha = jnp.maximum(jnp.max(jnp.abs(u)), T_N / (t_final - t))
        u_next = alpha * _step_raw(u / alpha)     # normalise, bloc du modèle, remise à l'échelle
        return u_next, t + T_N / alpha, n_blocks + 1

    u_pred, _, n_blocks = jax.lax.while_loop(cond_fn, body_fn, (u0, 0.0, 0))
    return u_pred, n_blocks


def _paired_block(u_model, u_true, t, t_final):
    """Un bloc adaptatif commun : durée fixée par l'état du modèle (alpha comme model_rollout,
    dernier bloc raccourci sur t_final), le solveur avance u_true de la même durée.
    Une fois t_final atteint (à 1e-6 T_N près, arrondis float32), tout est figé : sinon alpha
    exploserait et amplifierait le bruit d'arrondi du réseau sur u / alpha ~ 0."""
    T_N = T_of(u_model.shape[-1])
    done = t_final - t <= 1e-6 * T_N
    alpha = jnp.maximum(V_max(u_model), T_N / jnp.maximum(t_final - t, 1e-6 * T_N))
    T_k = jnp.where(done, 0.0, T_N / alpha)
    u_model_new = jnp.where(done, u_model, alpha * _step_raw(u_model / alpha))
    u_true_new  = jnp.where(done, u_true, _active_solver(u_true, T_k)[0])
    return u_model_new, u_true_new, t + T_k


@partial(jax.jit, static_argnames=("n_steps",))
def error_growth(u0, n_steps, t_final):
    """Croissance de l'erreur bloc par bloc, blocs adaptatifs jusqu'à t_final.

    On avance en parallèle deux trajectoires depuis le même u0, sur les mêmes durées de bloc :
      - u_true  : le vrai schéma numérique (référence) ;
      - u_model : le modèle.
    À chaque bloc on mesure la norme L2 physique de (u_model − u_true) et de u_true.
    n_steps = n_steps_of(t_final, N) majore le nombre de blocs (V <= 1 -> blocs >= T_N) ;
    une fois t_final atteint, les blocs suivants sont de durée ~0.

    Renvoie (errs, norms_true, times), chacun de longueur n_steps.
    """
    def block(carry, _):
        u_model, u_true, t = carry
        u_model, u_true, t = _paired_block(u_model, u_true, t, t_final)
        return (u_model, u_true, t), (l2(u_model - u_true), l2(u_true), t)

    _, (errs, norms_true, times) = jax.lax.scan(block, (u0, u0, 0.0), None, length=n_steps)
    return errs, norms_true, times


@partial(jax.jit, static_argnames=("n_steps",))
def trajectories(u0, n_steps, t_final):
    """Comme error_growth, mais renvoie les champs à chaque bloc :
    (u_true, u_model, times), champs de forme (n_steps + 1, N), u0 et t = 0 inclus."""
    def block(carry, _):
        u_model, u_true, t = carry
        u_model, u_true, t = _paired_block(u_model, u_true, t, t_final)
        return (u_model, u_true, t), (u_true, u_model, t)

    _, (u_true, u_model, times) = jax.lax.scan(block, (u0, u0, 0.0), None, length=n_steps)
    return (jnp.concatenate([u0[None], u_true], axis=0),
            jnp.concatenate([u0[None], u_model], axis=0),
            jnp.concatenate([jnp.zeros(1), times]))


def _bench(fn, *args, warmup=2, repeat=5):
    """Chronométrage propre : `warmup` appels non mesurés pour absorber la
    compilation XLA, puis le minimum de `repeat` appels mesurés."""
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


# Fonctions initiales absentes du dataset d'entraînement, échantillonnées sur
# chaque grille : c'est la même fonction physique u0(x) à toutes les résolutions
# (cf. "Limite" du doc §2 : en cellules, elle paraît étirée quand N grandit).
def make_test_functions(x):
    u0_somme_sinus = (
        1.0 * jnp.sin(2 * jnp.pi * 1 * x)
        + 0.5 * jnp.sin(2 * jnp.pi * 3 * x + 0.7)
        + 0.3 * jnp.sin(2 * jnp.pi * 7 * x + 2.1)
    )
    return {
        "triangle":     2 / jnp.pi * jnp.arcsin(jnp.sin(2 * jnp.pi * x)),
        "carre":        jnp.sign(jnp.sin(2 * jnp.pi * x)),
        "paquet_onde":  jnp.exp(-100 * (x - 0.5) ** 2) * jnp.sin(8 * jnp.pi * x),
        "sinus_simple": jnp.sin(2 * jnp.pi * x),
        "somme_sinus":  u0_somme_sinus / jnp.max(jnp.abs(u0_somme_sinus)),
        # Riemann simple (formation de choc en Burgers)
        "riemann":      jnp.where(x < 0.5, jnp.sin(70*jnp.pi*x), jnp.exp(-100 * (x - 0.5) ** 2) * jnp.sin(8 * jnp.pi * x)),
    }


grids          = {N: make_grid(N) for N in RESOLUTIONS}
test_functions = {N: make_test_functions(grids[N]) for N in RESOLUTIONS}
FUNCTION_NAMES = list(test_functions[RESOLUTIONS[0]])

# Temps physiques fixes : n_steps = round(t / T_N) dépend de la grille.
PHYSICAL_TIMES = [T]   # T de network_parameters ; ex. [0.1, 1, T] pour plusieurs horizons

print(f"lambda = {LAMBDA:g} fixé ; " + ", ".join(
    f"N={N} : T={T_of(N):g}, {make_grid(N).shape[0]} points" for N in RESOLUTIONS))

# ------------------------------------------------------------------
# Snapshots : cible vs modèle après n_blocks blocs, un
# panneau par résolution. Un bloc dure T_N = LAMBDA / N : n_blocks = 1 donne
# N=256 à T=0.1, N=512 à T=0.05, N=1024 à T=0.025 (même lambda, même nombre
# de pas -> c'est la comparaison directe de l'invariance du doc §2).
# ------------------------------------------------------------------
SNAPSHOT_BLOCKS = [1]   # nombres de blocs (ex: [1, 10, 100])
SNAPSHOT_FUNCTIONS = ["triangle", "carre", "somme_sinus", "sinus_simple", "riemann"]


def make_snapshot(name, n_steps):
    fig, axes = plt.subplots(1, len(RESOLUTIONS), figsize=(6 * len(RESOLUTIONS), 5),
                             sharey=True, squeeze=False)
    for ax, N in zip(axes[0], RESOLUTIONS):
        x, u0 = grids[N], test_functions[N][name]

        u_true = solver_rollout(u0, n_steps)
        u_pred, _ = model_rollout(u0, n_steps * T_of(N))   # même temps physique que la cible

        ax.plot(x, u0, label='u₀', linestyle='--', alpha=0.4, color='gray')
        ax.plot(x, u_true, label='cible', linewidth=1.5)
        ax.plot(x, u_pred, label='prédit', linewidth=1.5, linestyle=':')
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"N={N}, T={T_of(N):g}, {n_steps} pas  (t={n_steps * T_of(N):g})")
    axes[0][0].legend(loc='upper right')
    fig.suptitle(f"« {name} »  —  lambda={LAMBDA:g}")
    fig.tight_layout()
    out_path = f"snapshot_{name}_{n_steps}blocs.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Snapshot sauvegardé : {out_path}")


for name in SNAPSHOT_FUNCTIONS:
    for n_blocks in SNAPSHOT_BLOCKS:
        make_snapshot(name, n_blocks)

# ------------------------------------------------------------------
# Films .mp4 : évolution de la cible et du modèle bloc par bloc jusqu'à
# MOVIE_TIME, un film par (fonction, résolution). Nécessite ffmpeg
# (binaire système, ou `pip install imageio-ffmpeg`).
# ------------------------------------------------------------------
MAKE_MOVIES = True     # False pour désactiver les films
MOVIE_TIME = T         # temps physique final du film (T de network_parameters)
MOVIE_FPS = 20
MOVIE_FUNCTIONS = ["sinus_simple","riemann"]
MOVIE_RESOLUTIONS = [RESOLUTIONS[0]]   # films coûteux : par défaut la grille d'entraînement seule


def _movie_writer():
    import matplotlib.animation as animation
    if not animation.FFMpegWriter.isAvailable():
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError("ffmpeg introuvable : installer ffmpeg ou `pip install imageio-ffmpeg`")
    return animation.FFMpegWriter(fps=MOVIE_FPS, bitrate=2000)


def make_movie(name, N, t_final):
    import numpy as np
    from matplotlib.animation import FuncAnimation

    x, u0 = grids[N], test_functions[N][name]
    n_steps = n_blocks_max(t_final, N)
    u_true, u_pred, times = map(np.asarray, trajectories(u0, n_steps, t_final))
    n_frames = int(np.argmax(times >= t_final - 1e-6 * T_of(N))) + 1   # coupe les blocs figés après t_final
    u_true, u_pred, times = u_true[:n_frames], u_pred[:n_frames], times[:n_frames]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, u0, label='u₀', linestyle='--', alpha=0.4, color='gray')
    line_true, = ax.plot(x, u_true[0], label='cible', linewidth=1.5)
    line_pred, = ax.plot(x, u_pred[0], label='prédit', linewidth=1.5, linestyle=':')
    # Échelle verticale fixée sur la cible (le modèle pur peut diverger).
    lo, hi = float(u_true.min()), float(u_true.max())
    pad = 0.15 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    title = ax.set_title("")
    fig.tight_layout()

    def update(k):
        line_true.set_ydata(u_true[k])
        line_pred.set_ydata(u_pred[k])
        title.set_text(f"« {name} »  —  N={N}  t={times[k]:.2f}  (bloc {k}, T adaptatif)")
        return [line_true, line_pred, title]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out_path = f"movie_{name}_N{N}_t{t_final:g}.mp4"
    anim.save(out_path, writer=_movie_writer(), dpi=120)
    plt.close(fig)
    print(f"Film sauvegardé : {out_path}")


if MAKE_MOVIES:
    for N in MOVIE_RESOLUTIONS:
        for name in MOVIE_FUNCTIONS:
            make_movie(name, N, MOVIE_TIME)

    # Films supplémentaires aux grilles fines (T_N = 0.05 à N=512, 0.025 à N=1024).
    EXTRA_MOVIE_FUNCTIONS = ["riemann"]
    EXTRA_MOVIE_RESOLUTIONS = [N for N in (512, 1024) if N in RESOLUTIONS]   # vide si on reste à 256
    for N in EXTRA_MOVIE_RESOLUTIONS:
        for name in EXTRA_MOVIE_FUNCTIONS:
            make_movie(name, N, MOVIE_TIME)

if SOLVER != "advection":
    from burgers_solver import flux as _burgers_flux

    def burgers_step_cfl(u):
        "Un pas Godunov, dt = cfl*dx/max|u|, jamais clampé sur T."
        dx = 1.0 / u.shape[-1]
        dt = cfl * dx / (jnp.max(jnp.abs(u)) + 1e-10)
        f_face = _burgers_flux(u, jnp.roll(u, -1))
        u_new = u - dt / dx * (f_face - jnp.roll(f_face, 1))
        return u_new, dt

    @jax.jit
    def burgers_rollout_cfl(u0, t_final):
        "Solveur pas-à-pas classique : avance en pas dt naturels jusqu'à t_final."
        def cond_fn(carry):
            u, t, n_dt = carry
            return t < t_final

        def body_fn(carry):
            u, t, n_dt = carry
            u_new, dt = burgers_step_cfl(u)
            return (u_new, t + dt, n_dt + 1)

        return jax.lax.while_loop(cond_fn, body_fn, (u0, 0.0, 0))

    print("\n--- Perf : modèle (blocs CFL adaptatifs) vs solveur pas-à-pas ---")
    for N in PERF_RESOLUTIONS:
        perf_functions = test_functions.get(N) or make_test_functions(make_grid(N))
        for name in FUNCTION_NAMES:
            u0 = perf_functions[name]
            for t_phys in PHYSICAL_TIMES:
                (u_model, n_steps_model), t_model = _bench(model_rollout, u0, t_phys)
                (u_solver, _, n_dt_steps), t_solver = _bench(burgers_rollout_cfl, u0, t_phys)

                mse = float(jnp.mean((u_model - u_solver) ** 2))
                mse_norm = mse / (float(jnp.mean(u_solver ** 2)) + 1e-12)
                speedup = t_solver / t_model if t_model > 0 else float('nan')
                print(f"[N={N}] [{name}] t={t_phys:g}  modèle={int(n_steps_model)} blocs ({t_model*1e3:.2f}ms)  "
                      f"solveur={int(n_dt_steps)} pas dt ({t_solver*1e3:.2f}ms)  MSE={mse:.6f}  "
                      f"MSE_norm={mse_norm:.6f}  (×{speedup:.1f})")

# ------------------------------------------------------------------
# Erreur d'énergie par bande de fréquence, en fonction de l'horizon de
# rollout.
#   E_B(t)   = Σ_{k∈bande} |û(k,t)|²
#   ε_B(t)   = |E_B,pred(t) − E_B,true(t)| / (E_B,true(t) + ε)
# Les bandes sont en nombre d'onde PHYSIQUE k (le même sur toutes les
# grilles) : low k∈[1,5], mid k∈[6,15], hi1 k∈[16,31], hi2 k∈[32,47],
# hi3 k∈[48,63], hi4 k∈[64,95], hi5 k∈[96,128]. Au-delà de k=128 (modes
# qui n'existent qu'à N>256) : bande "fine", k∈[129, N/2].
# ε_B est un rapport, donc insensible à la normalisation de la rfft (∝ N²).
# ------------------------------------------------------------------
FREQ_BANDS = {
    "low": (1, 5), "mid": (6, 15), "hi1": (16, 31), "hi2": (32, 47),
    "hi3": (48, 63), "hi4": (64, 95), "hi5": (96, 128), "fine": (129, None),
}
EPS_BAND = 1e-12


def band_energy(u):
    "E_B = Σ_k |û(k)|² pour k dans chaque bande de FREQ_BANDS (bande vide -> 0)."
    power = jnp.abs(jnp.fft.rfft(u, axis=-1)) ** 2
    k_last = power.shape[-1] - 1
    return {name: jnp.sum(power[..., k0:(k_last if k1 is None else k1) + 1], axis=-1)
            for name, (k0, k1) in FREQ_BANDS.items()}


def band_error(u_pred, u_true):
    "ε_B par bande, entre un champ prédit et sa référence."
    e_pred, e_true = band_energy(u_pred), band_energy(u_true)
    return {name: jnp.abs(e_pred[name] - e_true[name]) / (e_true[name] + EPS_BAND) for name in FREQ_BANDS}


def print_band_table(title):
    print(f"\n--- {title} ---")
    print(f"{'N':>6}{'traj':>14}{'t':>8}{'n_steps':>9}" + "".join(f"{b:>11}" for b in FREQ_BANDS))
    for N in RESOLUTIONS:
        for name in FUNCTION_NAMES:
            u0 = test_functions[N][name]
            for t_phys in PHYSICAL_TIMES:
                n_steps = n_steps_of(t_phys, N)
                u_true = solver_rollout(u0, n_steps)
                u_pred, _ = model_rollout(u0, n_steps * T_of(N))   # même temps physique que la cible
                errs = band_error(u_pred, u_true)
                print(f"{N:>6}{name:>14}{t_phys:>8g}{n_steps:>9}"
                      + "".join(f"{float(errs[b]):>11.2e}" for b in FREQ_BANDS))


print_band_table("Erreur par bande de fréquence")

# ------------------------------------------------------------------
# Croissance de l'erreur ‖û_n − u_n(vrai)‖_L2 en fonction du temps physique
# t_n = n · T_N : un panneau par fonction test, une courbe par résolution.
# Si l'invariance tient, les courbes se superposent (à nombre de pas
# double/quadruple près pour atteindre le même t, cf. doc §8).
# ------------------------------------------------------------------
# None -> temps physique max des figures de prédiction (max(PHYSICAL_TIMES))
ERROR_GROWTH_TIME = 1
error_growth_time = ERROR_GROWTH_TIME if ERROR_GROWTH_TIME is not None else max(PHYSICAL_TIMES)

n_cols = 3
n_rows = -(-len(FUNCTION_NAMES) // n_cols)
fig_err, axes_err = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
for ax, name in zip(axes_err.flat, FUNCTION_NAMES):
    for N in RESOLUTIONS:
        n_steps_err = n_blocks_max(error_growth_time, N)
        errs_pur, norms_true, t_n = error_growth(test_functions[N][name], n_steps_err, error_growth_time)
        n_bl = int(jnp.argmax(t_n >= error_growth_time - 1e-6 * T_of(N))) + 1   # blocs utiles
        # erreur L2 relative au carré, même quantité que MSE_norm (perf) : ‖û − u‖² / ‖u‖²
        #errs_rel = (errs_pur[:n_bl] / (norms_true[:n_bl] + 1e-12)) ** 2
        t_n = t_n[:n_bl]
        err_abs  = errs_pur[:n_bl] ** 2      # ‖û − u‖² : erreur absolue, sans normalisation
        #line, = ax.plot(t_n, errs_rel, linewidth=1.5, marker=".", label=f"N={N} avec norm ({n_bl} blocs)")
        ax.plot(t_n, err_abs, linewidth=1.5, marker=".", label=f"N={N} ({n_bl} blocs)")
        #print(f"[{name}] N={N} ‖ε‖²/‖u‖² : t={float(t_n[0]):.3g} → {float(errs_rel[0]):.3e}   "
        #      f"t={float(t_n[-1]):g} → {float(errs_rel[-1]):.3e}   ({n_bl} blocs)")
    ax.set_title(name)
    ax.set_xlabel("t (blocs adaptatifs T_N / max|u|)")
    ax.set_ylabel("‖û−u‖²   (L2)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)
for ax in list(axes_err.flat)[len(FUNCTION_NAMES):]:
    ax.axis("off")
fig_err.suptitle(f"Croissance de l'erreur du modèle à lambda={LAMBDA:g} fixé")
fig_err.tight_layout()
fig_err.savefig("error_growth.png", dpi=150)
print("Figure sauvegardée : error_growth.png")

plt.show()
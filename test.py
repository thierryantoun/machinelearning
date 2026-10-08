import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from network_parameters import SOLVER, T_target, n, cfl
from loss import predict_F

if SOLVER == "advection":
    from advection_solver import advection_solver as _active_solver
else:
    from burgers_solver import burgers_solver as _active_solver

with open("params_fno.pkl", "rb") as f:
    params = pickle.load(f)

# ------------------------------------------------------------------
# Évaluation multi-résolution à lambda = T / dx fixé (doc §2, protocole
# étape 2) : le modèle est entraîné à n = N0, puis évalué SANS
# réentraînement sur d'autres grilles avec T_N = LAMBDA / N.
# Domaine périodique [0, 1) -> dx = 1 / N. Les mêmes P, Q (fenêtre en
# cellules) servent sur toutes les grilles ; le nombre de modes du FNO
# suit N (K = kappa N / 2, R interpolé, cf. network.py).
#
# Tout ce qui dépend de la grille (dx, T, nombre de pas) est déduit de la
# taille de u : les fonctions jit ci-dessous se recompilent une fois par
# résolution, rien d'autre à changer.
# ------------------------------------------------------------------
LAMBDA      = float(T_target * n)   # 25.6 à l'entraînement (n=256, T=0.1)
RESOLUTIONS = [256, 512, 1024]


def make_grid(N):
    return jnp.linspace(0, 1, N, endpoint=False)


def T_of(N):
    "Pas de temps d'un bloc sur la grille N, à lambda fixé."
    return LAMBDA / N


def n_steps_of(t_phys, N):
    return max(1, round(t_phys / T_of(N)))


def solver(u):
    "Un bloc du vrai schéma, sur le temps T_N de la grille de u."
    return _active_solver(u, T_of(u.shape[-1]))


def step(u):
    "Un bloc du modèle : u - lambda (F_{j+1/2} - F_{j-1/2}), lambda = T_N / dx identique sur toute grille."
    F = predict_F(params, u)
    return u - LAMBDA * (F - jnp.roll(F, 1, axis=-1))


def l2(v):
    "Norme L2 physique sqrt(dx * sum v^2), dx = 1/N : comparable d'une grille à l'autre."
    return jnp.sqrt(jnp.sum(v ** 2, axis=-1) / v.shape[-1])


@partial(jax.jit, static_argnames=("n_steps",))
def solver_rollout(u0, n_steps):
    def solver_block(u, _):
        return solver(u)[0], None
    u_target, _ = jax.lax.scan(solver_block, u0, None, length=n_steps)
    return u_target


@partial(jax.jit, static_argnames=("n_steps", "correction_every"))
def model_rollout(u0, n_steps, correction_every=None):
    """Rollout du modèle. Si correction_every est donné, tous les
    `correction_every` blocs la solution est réinjectée un pas dans le vrai
    schéma numérique à la place de la prédiction du modèle."""
    def model_block(u, i):
        u_next = step(u)
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
                  tous les `correction_every` blocs.
    À chaque bloc on mesure la norme L2 physique de (u_model − u_true),
    (u_corr − u_true) et de u_true.

    Renvoie (errs_pur, errs_corr, norms_true), chacun de longueur n_steps.
    """
    def block(carry, i):
        u_model, u_corr, u_true = carry
        u_true_next = solver(u_true)[0]
        u_model_next = step(u_model)
        u_corr_next = step(u_corr)
        if correction_every:
            do_correct = ((i + 1) % correction_every) == 0
            u_corr_next = jax.lax.cond(
                do_correct, lambda uu: solver(uu)[0], lambda uu: uu, u_corr_next
            )
        errs = (l2(u_model_next - u_true_next), l2(u_corr_next - u_true_next), l2(u_true_next))
        return (u_model_next, u_corr_next, u_true_next), errs

    _, (errs_pur, errs_corr, norms_true) = jax.lax.scan(
        block, (u0, u0, u0), jnp.arange(n_steps)
    )
    return errs_pur, errs_corr, norms_true


@partial(jax.jit, static_argnames=("n_steps", "correction_every"))
def trajectories(u0, n_steps, correction_every=None):
    """Comme error_growth, mais renvoie les champs à chaque bloc :
    (u_true, u_model, u_corr), chacun de forme (n_steps + 1, N), u0 inclus."""
    def block(carry, i):
        u_true, u_model, u_corr = carry
        u_true = solver(u_true)[0]
        u_model = step(u_model)
        u_corr = step(u_corr)
        if correction_every:
            do_correct = ((i + 1) % correction_every) == 0
            u_corr = jax.lax.cond(do_correct, lambda uu: solver(uu)[0], lambda uu: uu, u_corr)
        return (u_true, u_model, u_corr), (u_true, u_model, u_corr)

    _, trajs = jax.lax.scan(block, (u0, u0, u0), jnp.arange(n_steps))
    return tuple(jnp.concatenate([u0[None], tr], axis=0) for tr in trajs)


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
        "riemann":      jnp.where(x < 0.5, 1.0, -0.5),
    }


grids          = {N: make_grid(N) for N in RESOLUTIONS}
test_functions = {N: make_test_functions(grids[N]) for N in RESOLUTIONS}
FUNCTION_NAMES = list(test_functions[RESOLUTIONS[0]])

# Temps physiques fixes : n_steps = round(t / T_N) dépend de la grille.
PHYSICAL_TIMES = [0.1, 1, 5, 10, 200]

print(f"lambda = {LAMBDA:g} fixé ; " + ", ".join(
    f"N={N} : T={T_of(N):g}, {make_grid(N).shape[0]} points" for N in RESOLUTIONS))

# ------------------------------------------------------------------
# Erreur à UN pas, par résolution (protocole étape 2). Prédiction du doc :
# erreur par pas du même ordre aux trois résolutions, pas d'explosion en
# x = 0.5. On affiche l'erreur relative L2 (comme la loss) et l'endroit de
# l'erreur ponctuelle max.
# ------------------------------------------------------------------
print("\n--- Erreur à un pas (relative L2 | max |err| @ x) ---")
print(f"{'traj':<14}" + "".join(f"{'N=' + str(N):>30}" for N in RESOLUTIONS))
for name in FUNCTION_NAMES:
    row = f"{name:<14}"
    for N in RESOLUTIONS:
        u0 = test_functions[N][name]
        u_true = solver(u0)[0]
        u_pred = step(u0)
        err = u_pred - u_true
        rel = float(l2(err) / (l2(u_true) + 1e-12))
        j = int(jnp.argmax(jnp.abs(err)))
        row += f"{rel:>12.2e} | {float(jnp.abs(err[j])):.2e} @ {float(grids[N][j]):.3f}"
    print(row)

# ------------------------------------------------------------------
# Test d'exactitude de l'invariance (doc §2) : sur la grille m*n, le champ
# u0(m x) échantillonné vaut exactement u0 sur la grille n répété m fois
# (jnp.tile, u0 périodique de période 1). Toutes les opérations étant en
# cellules (fenêtres, couches ponctuelles, R interpolé qui tombe exactement
# sur la table aux modes multiples de m), F, le pas du modèle et le pas du
# solveur doivent être ceux de la grille n, répétés m fois, à l'arrondi
# float32 près (~1e-6). Un écart plus grand = bug d'implémentation, pas un
# défaut de généralisation.
# ------------------------------------------------------------------
EXACT_TOL = 1e-4

print(f"\n--- Test d'exactitude : u0(m x) sur N = m*{n}  vs  u0(x) sur N = {n} répété m fois ---")
print("    (écart max absolu ; attendu ~1e-6)")
print(f"{'traj':<14}{'N':>6}{'F':>12}{'pas modèle':>12}{'pas solveur':>13}")
exact_ok = True
for name in FUNCTION_NAMES:
    u0_ref = make_test_functions(make_grid(n))[name]
    F_ref, u_model_ref, u_true_ref = predict_F(params, u0_ref), step(u0_ref), solver(u0_ref)[0]
    for N in RESOLUTIONS:
        if N == n or N % n != 0:
            continue
        m = N // n
        u0_m = jnp.tile(u0_ref, m)
        gaps = [float(jnp.max(jnp.abs(a - jnp.tile(b, m)))) for a, b in (
            (predict_F(params, u0_m), F_ref),
            (step(u0_m), u_model_ref),
            (solver(u0_m)[0], u_true_ref),
        )]
        exact_ok &= max(gaps) < EXACT_TOL
        print(f"{name:<14}{N:>6}" + "".join(f"{g:>12.2e}" for g in gaps[:2]) + f"{gaps[2]:>13.2e}")
print("=> invariance exacte OK" if exact_ok else
      f"=> ÉCART > {EXACT_TOL:g} : l'implémentation n'est pas exactement en unités de cellules")

# ------------------------------------------------------------------
# Riemann vu EN CELLULES autour du saut initial (x = 0.5).
# Un saut est un saut sur une cellule à toute résolution : à lambda fixé,
# les fenêtres autour du choc sont identiques cellule par cellule à
# 256, 512 et 1024 points (l'autre saut, en x = 0, est à >= 128 cellules,
# hors de la fenêtre de P+Q+1 cellules). La partie LOCALE du réseau rend
# donc la même sortie partout ; un écart entre résolutions ne peut venir
# que de la partie non locale (branche spectrale R).
#   - kmax = 0 : on attend un écart ~1e-6 (float32).
#   - kmax > 0 : l'écart mesure l'effet longue portée de R.
# ------------------------------------------------------------------
CELL_HALF_WIDTH = 40          # cellules affichées de part et d'autre du saut
CELL_FUNCTIONS  = ["riemann"]

jj = jnp.arange(-CELL_HALF_WIDTH, CELL_HALF_WIDTH + 1)
print(f"\n--- Riemann en cellules : écart max à N={RESOLUTIONS[0]}, sur ±{CELL_HALF_WIDTH} cellules du saut ---")
print(f"    (kmax = 0 -> attendu ~1e-6 ; sinon : effet longue portée de la branche spectrale)")
for name in CELL_FUNCTIONS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ref_pred = ref_true = None
    for N in RESOLUTIONS:
        u0 = test_functions[N][name]
        j0 = N // 2                                   # saut initial en x = 0.5
        up = step(u0)[j0 + jj]
        ut = solver(u0)[0][j0 + jj]
        axes[0].plot(jj, up, "o-", ms=3, lw=1, label=f"modèle N={N}")
        if N == RESOLUTIONS[0]:
            axes[0].plot(jj, ut, "k--", lw=1.2, label=f"cible N={N}")
            ref_pred, ref_true = up, ut
        else:
            axes[1].plot(jj, up - ref_pred, "o-", ms=3, lw=1, label=f"modèle N={N} − N={RESOLUTIONS[0]}")
            axes[1].plot(jj, ut - ref_true, "--", lw=1, label=f"cible N={N} − N={RESOLUTIONS[0]}")
            print(f"[{name}] N={N} : modèle {float(jnp.max(jnp.abs(up - ref_pred))):.2e}   "
                  f"cible {float(jnp.max(jnp.abs(ut - ref_true))):.2e}")
    axes[0].set_title(f"« {name} » après 1 pas, en cellules (lambda={LAMBDA:g})")
    axes[1].set_title(f"écart à N={RESOLUTIONS[0]}, cellule par cellule")
    for ax in axes:
        ax.set_xlabel("cellules depuis le saut initial (x = 0.5)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = f"cellules_{name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")

# ------------------------------------------------------------------
# Snapshots : cible vs modèle (pur et corrigé) après n_blocks blocs, un
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
        has_corr = CORRECTION_EVERY > 0 and n_steps >= CORRECTION_EVERY

        u_true = solver_rollout(u0, n_steps)
        u_pred = model_rollout(u0, n_steps)
        u_corr = model_rollout(u0, n_steps, correction_every=CORRECTION_EVERY) if has_corr else None

        ax.plot(x, u0, label='u₀', linestyle='--', alpha=0.4, color='gray')
        ax.plot(x, u_true, label='cible', linewidth=1.5)
        ax.plot(x, u_pred, label='prédit (pur)', linewidth=1.5, linestyle=':')
        if has_corr:
            ax.plot(x, u_corr, label=f'prédit (corrigé/{CORRECTION_EVERY})', linewidth=1.5, linestyle='-.')
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
MOVIE_TIME = 10.0          # temps physique final du film
MOVIE_FPS = 20
MOVIE_FUNCTIONS = ["triangle", "carre", "somme_sinus", "sinus_simple", "riemann"]
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
    T_N = T_of(N)
    n_steps = n_steps_of(t_final, N)
    has_corr = CORRECTION_EVERY > 0 and n_steps >= CORRECTION_EVERY
    u_true, u_pred, u_corr = map(np.asarray, trajectories(
        u0, n_steps, correction_every=CORRECTION_EVERY if has_corr else None))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, u0, label='u₀', linestyle='--', alpha=0.4, color='gray')
    line_true, = ax.plot(x, u_true[0], label='cible', linewidth=1.5)
    line_pred, = ax.plot(x, u_pred[0], label='prédit (pur)', linewidth=1.5, linestyle=':')
    line_corr = None
    if has_corr:
        line_corr, = ax.plot(x, u_corr[0], label=f'prédit (corrigé/{CORRECTION_EVERY})',
                             linewidth=1.5, linestyle='-.')
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
        if line_corr is not None:
            line_corr.set_ydata(u_corr[k])
        title.set_text(f"« {name} »  —  N={N}  t={k * T_N:.2f}  (T={T_N:g})")
        artists = [line_true, line_pred, title]
        return artists + ([line_corr] if line_corr is not None else [])

    anim = FuncAnimation(fig, update, frames=n_steps + 1, blit=False)
    out_path = f"movie_{name}_N{N}_t{t_final:g}.mp4"
    anim.save(out_path, writer=_movie_writer(), dpi=120)
    plt.close(fig)
    print(f"Film sauvegardé : {out_path}")


for N in MOVIE_RESOLUTIONS:
    for name in MOVIE_FUNCTIONS:
        make_movie(name, N, MOVIE_TIME)

# Films supplémentaires aux grilles fines (T_N = 0.05 à N=512, 0.025 à N=1024).
EXTRA_MOVIE_FUNCTIONS = ["riemann", "carre"]
EXTRA_MOVIE_RESOLUTIONS = [512, 1024]
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

    print("\n--- Perf : modèle (blocs T_N) vs solveur pas-à-pas ---")
    for N in RESOLUTIONS:
        for name in FUNCTION_NAMES:
            u0 = test_functions[N][name]
            for t_phys in PHYSICAL_TIMES:
                n_steps_model = n_steps_of(t_phys, N)

                u_model, t_model = _bench(model_rollout, u0, n_steps_model)
                (u_solver, _, n_dt_steps), t_solver = _bench(burgers_rollout_cfl, u0, t_phys)

                mse = float(jnp.mean((u_model - u_solver) ** 2))
                mse_norm = mse / (float(jnp.mean(u_solver ** 2)) + 1e-12)
                speedup = t_solver / t_model if t_model > 0 else float('nan')
                print(f"[N={N}] [{name}] t={t_phys:g}  modèle={n_steps_model} blocs ({t_model*1e3:.2f}ms)  "
                      f"solveur={int(n_dt_steps)} pas dt ({t_solver*1e3:.2f}ms)  MSE={mse:.6f}  "
                      f"MSE_norm={mse_norm:.6f}  (×{speedup:.1f})")

# ------------------------------------------------------------------
# Erreur d'énergie par bande de fréquence, en fonction de l'horizon de
# rollout, avec et sans correction.
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


def print_band_table(title, correction_every):
    print(f"\n--- {title} ---")
    print(f"{'N':>6}{'traj':>14}{'t':>8}{'n_steps':>9}" + "".join(f"{b:>11}" for b in FREQ_BANDS))
    for N in RESOLUTIONS:
        for name in FUNCTION_NAMES:
            u0 = test_functions[N][name]
            for t_phys in PHYSICAL_TIMES:
                n_steps = n_steps_of(t_phys, N)
                u_true = solver_rollout(u0, n_steps)
                u_pred = model_rollout(u0, n_steps, correction_every=correction_every)
                errs = band_error(u_pred, u_true)
                print(f"{N:>6}{name:>14}{t_phys:>8g}{n_steps:>9}"
                      + "".join(f"{float(errs[b]):>11.2e}" for b in FREQ_BANDS))


print_band_table("Erreur par bande de fréquence — SANS correction", correction_every=None)
if CORRECTION_EVERY > 0:
    print_band_table(f"Erreur par bande de fréquence — AVEC correction (every={CORRECTION_EVERY})",
                     correction_every=CORRECTION_EVERY)

# ------------------------------------------------------------------
# Croissance de l'erreur ‖û_n − u_n(vrai)‖_L2 en fonction du temps physique
# t_n = n · T_N : un panneau par fonction test, une courbe par résolution.
# Si l'invariance tient, les courbes se superposent (à nombre de pas
# double/quadruple près pour atteindre le même t, cf. doc §8).
# ------------------------------------------------------------------
# None -> temps physique max des figures de prédiction (max(PHYSICAL_TIMES))
ERROR_GROWTH_TIME = None
error_growth_time = ERROR_GROWTH_TIME if ERROR_GROWTH_TIME is not None else max(PHYSICAL_TIMES)

n_cols = 3
n_rows = -(-len(FUNCTION_NAMES) // n_cols)
fig_err, axes_err = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
for ax, name in zip(axes_err.flat, FUNCTION_NAMES):
    for N in RESOLUTIONS:
        n_steps_err = n_steps_of(error_growth_time, N)
        t_n = jnp.arange(1, n_steps_err + 1) * T_of(N)
        errs_pur, _, _ = error_growth(test_functions[N][name], n_steps_err)
        ax.plot(t_n, errs_pur, linewidth=1.5, label=f"N={N}")
        print(f"[{name}] N={N} ‖ε‖_L2 : t={T_of(N):g} → {float(errs_pur[0]):.3e}   "
              f"t={float(t_n[-1]):g} → {float(errs_pur[-1]):.3e}")
    ax.set_title(name)
    ax.set_xlabel("t_n = n · T_N")
    ax.set_ylabel("‖û_n − u_n(vrai)‖_L2")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
for ax in list(axes_err.flat)[len(FUNCTION_NAMES):]:
    ax.axis("off")
fig_err.suptitle(f"Croissance de l'erreur du modèle à lambda={LAMBDA:g} fixé")
fig_err.tight_layout()
fig_err.savefig("error_growth.png", dpi=150)
print("Figure sauvegardée : error_growth.png")

plt.show()
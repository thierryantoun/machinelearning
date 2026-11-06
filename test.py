import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from network_parameters import MODEL, SOLVER, LAMBDA, n, cfl, T
from loss import predict_F, V_max, bloc, rollout
from euler_solver import euler_solver as _active_solver, flux, primitives, gamma

with open(f"params_{MODEL}_{SOLVER}.pkl", "rb") as f:
    params = pickle.load(f)

# Params entraînés avec l'ancienne version (noyau déclaré à la main : params/kernel, params/bias)
# -> rangés sous params/lifting pour le nn.Conv actuel. Même calcul, écart 0.
if "kernel" in params["params"]:
    p = dict(params["params"])
    p["lifting"] = {"kernel": p.pop("kernel"), "bias": p.pop("bias")}
    params = {"params": p}

# ------------------------------------------------------------------
# Évaluation multi-résolution à lambda = T * V / dx fixé (doc §2, protocole
# étape 2) : le modèle est entraîné à n = N0, puis évalué SANS
# réentraînement sur d'autres grilles avec T * V = LAMBDA / N.
# Domaine périodique [0, 1) -> dx = 1 / N. Les mêmes P, Q (fenêtre en
# cellules) servent sur toutes les grilles : le noyau de la conv (P+Q+1
# poids) est le même quelle que soit N.
#
# Tout ce qui dépend de la grille (dx, T, nombre de pas) est déduit de la
# taille de u : les fonctions jit ci-dessous se recompilent une fois par
# résolution, rien d'autre à changer.
# ------------------------------------------------------------------
RESOLUTIONS = [256]   # grille d'entraînement seule pour l'instant ; [256, 512, 1024] pour le multi-résolution
PERF_RESOLUTIONS = [8192]   # perf modèle vs solveur : grille d'entraînement + résolutions plus fines (λ fixe)


def make_grid(N):
    return jnp.linspace(0, 1, N, endpoint=False)


def T_of(N):
    "T * V d'un bloc sur la grille N, à lambda fixé : le bloc dure T_of(N) / V_max(u)."
    return LAMBDA / N


def n_steps_of(t_phys, u0):
    "Nombre de blocs estimé avec le V de u0 ; V change ensuite au fil des blocs."
    return max(1, round(t_phys * float(V_max(u0)) / T_of(u0.shape[-1])))


def solver(u):
    "Un bloc du vrai schéma, sur le temps T_of(N) / V_max(u) de la grille de u."
    return _active_solver(u, T_of(u.shape[-1]) / V_max(u))


def step(u):
    "Un bloc du modèle, à lambda = T * V / dx fixé."
    return bloc(params, u)[0]


def l2(v):
    "Norme L2 physique sqrt(dx * sum v^2), dx = 1/N, par variable : comparable d'une grille à l'autre."
    return jnp.sqrt(jnp.sum(v ** 2, axis=-1) / v.shape[-1])


@jax.jit
def solver_rollout(u0, t_final):
    "Le vrai schéma jusqu'au temps physique t_final."
    return _active_solver(u0, t_final)[0]


def model_rollout(u0, t_final=T):
    "Blocs du modèle jusqu'à t_final (loss.rollout). Renvoie (U(t_final), nombre de blocs)."
    return rollout(params, u0, t_final)


@partial(jax.jit, static_argnames=("n_steps",))
def error_growth(u0, n_steps):
    """Croissance de l'erreur bloc par bloc.

    On avance en parallèle deux trajectoires depuis le même u0 :
      - u_true  : le vrai schéma numérique (référence), sur la durée T_k du bloc du modèle,
        pour comparer les deux au même temps physique ;
      - u_model : le modèle.
    À chaque bloc on mesure la norme L2 physique de (u_model − u_true) et de u_true.

    Renvoie (errs, norms_true, ts) : (n_steps, 3), (n_steps, 3) et durées (n_steps,).
    """
    def block(carry, _):
        u_model, u_true = carry
        u_model_next, T_k = bloc(params, u_model)
        u_true_next = _active_solver(u_true, T_k)[0]
        errs = (l2(u_model_next - u_true_next), l2(u_true_next), T_k)
        return (u_model_next, u_true_next), errs

    _, (errs, norms_true, ts) = jax.lax.scan(block, (u0, u0), None, length=n_steps)
    return errs, norms_true, ts


@partial(jax.jit, static_argnames=("n_steps",))
def trajectories(u0, n_steps):
    """Comme error_growth, mais renvoie les champs à chaque bloc :
    (u_true, u_model), chacun de forme (n_steps + 1, 3, N), u0 inclus, et les durées ts."""
    def block(carry, _):
        u_true, u_model = carry
        u_model, T_k = bloc(params, u_model)
        u_true = _active_solver(u_true, T_k)[0]
        return (u_true, u_model), (u_true, u_model, T_k)

    _, (tr_true, tr_model, ts) = jax.lax.scan(block, (u0, u0), None, length=n_steps)
    return tuple(jnp.concatenate([u0[None], tr], axis=0) for tr in (tr_true, tr_model)), ts


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
def euler_state(rho, u, p):
    "(rho, u, p) -> variables conservatives (rho, m, E), (3, N)."
    return jnp.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2])


def make_test_functions(x):
    un = jnp.ones_like(x)
    sinus    = jnp.sin(2 * jnp.pi * x)
    triangle = 2 / jnp.pi * jnp.arcsin(sinus)
    paquet   = jnp.exp(-100 * (x - 0.5) ** 2) * jnp.sin(8 * jnp.pi * x)
    somme_sinus = (
        1.0 * sinus
        + 0.5 * jnp.sin(2 * jnp.pi * 3 * x + 0.7)
        + 0.3 * jnp.sin(2 * jnp.pi * 7 * x + 2.1)
    )
    somme_sinus = somme_sinus / jnp.max(jnp.abs(somme_sinus))
    return {
        "triangle":     euler_state(jnp.exp(0.5 * triangle), 0.5 * triangle, un),
        # contact pur : saut de densité transporté à u constant, p constant
        "carre":        euler_state(jnp.exp(0.5 * jnp.sign(sinus)), 0.5 * un, un),
        "paquet_onde":  euler_state(1.0 + 0.5 * paquet, 0.0 * un, 1.0 + 0.5 * paquet),
        "sinus_simple": euler_state(un, 0.5 * sinus, un),
        "somme_sinus":  euler_state(jnp.exp(0.5 * somme_sinus), 0.5 * somme_sinus, jnp.exp(0.5 * somme_sinus)),
        # Riemann type Sod (choc, contact, détente), densité dans la plage d'entraînement
        "riemann":      euler_state(jnp.where(x < 0.5, 2.5, 0.3125), 0.0 * un, jnp.where(x < 0.5, 1.0, 0.1)),
    }


grids          = {N: make_grid(N) for N in RESOLUTIONS}
test_functions = {N: make_test_functions(grids[N]) for N in RESOLUTIONS}
FUNCTION_NAMES = list(test_functions[RESOLUTIONS[0]])

# Temps physiques fixes : le nombre de blocs dépend de la grille et de l'état.
PHYSICAL_TIMES = [T]   # T de network_parameters ; ex. [0.1, 1, T] pour plusieurs horizons

print(f"lambda = {LAMBDA:g} fixé ; " + ", ".join(
    f"N={N} : T*V={T_of(N):g}, {make_grid(N).shape[0]} points" for N in RESOLUTIONS))

# ------------------------------------------------------------------
# Erreur à UN pas, par résolution (protocole étape 2). Prédiction du doc :
# erreur par pas du même ordre aux trois résolutions, pas d'explosion en
# x = 0.5. On affiche l'erreur relative L2 (comme la loss) et l'endroit de
# l'erreur ponctuelle max.
# ------------------------------------------------------------------
print("\n--- Erreur à un pas (relative L2 rho, m, E | max |err| @ x) ---")
print(f"{'traj':<14}" + "".join(f"{'N=' + str(N):>50}" for N in RESOLUTIONS))
for name in FUNCTION_NAMES:
    row = f"{name:<14}"
    for N in RESOLUTIONS:
        u0 = test_functions[N][name]
        u_true = solver(u0)[0]
        u_pred = step(u0)
        err = u_pred - u_true
        rel = l2(err) / (l2(u_true) + 1e-12)
        err_max = jnp.max(jnp.abs(err), axis=0)
        j = int(jnp.argmax(err_max))
        row += (f"{float(rel[0]):>10.2e} {float(rel[1]):>9.2e} {float(rel[2]):>9.2e}"
                f" | {float(err_max[j]):.2e} @ {float(grids[N][j]):.3f}")
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
# donc la même sortie partout : on attend un écart ~1e-6 (float32).
# ------------------------------------------------------------------
CELL_HALF_WIDTH = 40          # cellules affichées de part et d'autre du saut
CELL_FUNCTIONS  = ["riemann"]

jj = jnp.arange(-CELL_HALF_WIDTH, CELL_HALF_WIDTH + 1)
print(f"\n--- Riemann en cellules : écart max à N={RESOLUTIONS[0]}, sur ±{CELL_HALF_WIDTH} cellules du saut ---")
print(f"    (attendu ~1e-6 : réseau purement local)")
for name in CELL_FUNCTIONS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ref_pred = ref_true = None
    for N in RESOLUTIONS:
        u0 = test_functions[N][name]
        j0 = N // 2                                   # saut initial en x = 0.5
        up = step(u0)[0, j0 + jj]                     # rho
        ut = solver(u0)[0][0, j0 + jj]
        axes[0].plot(jj, up, "o-", ms=3, lw=1, label=f"modèle N={N}")
        if N == RESOLUTIONS[0]:
            axes[0].plot(jj, ut, "k--", lw=1.2, label=f"cible N={N}")
            ref_pred, ref_true = up, ut
        else:
            axes[1].plot(jj, up - ref_pred, "o-", ms=3, lw=1, label=f"modèle N={N} − N={RESOLUTIONS[0]}")
            axes[1].plot(jj, ut - ref_true, "--", lw=1, label=f"cible N={N} − N={RESOLUTIONS[0]}")
            print(f"[{name}] N={N} : modèle {float(jnp.max(jnp.abs(up - ref_pred))):.2e}   "
                  f"cible {float(jnp.max(jnp.abs(ut - ref_true))):.2e}")
    axes[0].set_title(f"rho de « {name} » après 1 pas, en cellules (lambda={LAMBDA:g})")
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
# Snapshots : cible vs modèle au temps de n_blocks blocs estimé avec le V de u0,
# une colonne par résolution, une ligne par variable (rho, u, p).
# Un bloc dure T_of(N) / V : même lambda, même nombre de pas à toute résolution
# -> c'est la comparaison directe de l'invariance du doc §2.
# ------------------------------------------------------------------
SNAPSHOT_BLOCKS = [10]   # nombres de blocs (ex: [1, 10, 100])
SNAPSHOT_FUNCTIONS = ["triangle", "carre", "somme_sinus", "sinus_simple", "riemann"]


def make_snapshot(name, n_steps):
    fig, axes = plt.subplots(3, len(RESOLUTIONS), figsize=(6 * len(RESOLUTIONS), 10),
                             sharex=True, squeeze=False)
    for col, N in enumerate(RESOLUTIONS):
        x, u0 = grids[N], test_functions[N][name]
        t_final = n_steps * T_of(N) / float(V_max(u0))

        u_true = solver_rollout(u0, t_final)
        u_pred, n_blocks = model_rollout(u0, t_final)   # même temps physique que la cible

        for row, (label, v0, v_true, v_pred) in enumerate(zip(
                ("rho", "u", "p"), primitives(u0), primitives(u_true), primitives(u_pred))):
            ax = axes[row][col]
            ax.plot(x, v0, label='état initial', linestyle='--', alpha=0.4, color='gray')
            ax.plot(x, v_true, label='cible', linewidth=1.5)
            ax.plot(x, v_pred, label='prédit', linewidth=1.5, linestyle=':')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[2][col].set_xlabel('x')
        axes[0][col].set_title(f"N={N}, {int(n_blocks)} blocs  (t={t_final:g})")
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
MAKE_MOVIES = False     # False pour désactiver les films
MOVIE_TIME = T         # temps physique final du film (T de network_parameters)
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
    n_steps = n_steps_of(t_final, u0)
    (u_true, u_pred), ts = trajectories(u0, n_steps)
    u_true, u_pred = np.asarray(u_true[:, 0]), np.asarray(u_pred[:, 0])   # rho, (n_steps + 1, N)
    t_k = np.concatenate([[0.0], np.cumsum(np.asarray(ts))])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, u0[0], label='rho initial', linestyle='--', alpha=0.4, color='gray')
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
        title.set_text(f"« {name} »  —  N={N}  t={t_k[k]:.3f}  (T*V={T_of(N):g})")
        return [line_true, line_pred, title]

    anim = FuncAnimation(fig, update, frames=n_steps + 1, blit=False)
    out_path = f"movie_{name}_N{N}_t{t_final:g}.mp4"
    anim.save(out_path, writer=_movie_writer(), dpi=120)
    plt.close(fig)
    print(f"Film sauvegardé : {out_path}")


if MAKE_MOVIES:
    for N in MOVIE_RESOLUTIONS:
        for name in MOVIE_FUNCTIONS:
            make_movie(name, N, MOVIE_TIME)

    # Films supplémentaires aux grilles fines (T*V = 0.05 à N=512, 0.025 à N=1024).
    EXTRA_MOVIE_FUNCTIONS = ["riemann", "carre"]
    EXTRA_MOVIE_RESOLUTIONS = [N for N in (512, 1024) if N in RESOLUTIONS]   # vide si on reste à 256
    for N in EXTRA_MOVIE_RESOLUTIONS:
        for name in EXTRA_MOVIE_FUNCTIONS:
            make_movie(name, N, MOVIE_TIME)

if SOLVER != "advection":
    def euler_step_cfl(U):
        "Un pas du schéma, dt = cfl*dx/V, jamais clampé sur T."
        dx = 1.0 / U.shape[-1]
        dt = cfl * dx / V_max(U)
        f_face = flux(U, jnp.roll(U, -1, axis=-1))
        U_new = U - dt / dx * (f_face - jnp.roll(f_face, 1, axis=-1))
        return U_new, dt

    @jax.jit
    def euler_rollout_cfl(u0, t_final):
        "Solveur pas-à-pas classique : avance en pas dt naturels jusqu'à t_final."
        def cond_fn(carry):
            u, t, n_dt = carry
            return t < t_final

        def body_fn(carry):
            u, t, n_dt = carry
            u_new, dt = euler_step_cfl(u)
            return (u_new, t + dt, n_dt + 1)

        return jax.lax.while_loop(cond_fn, body_fn, (u0, 0.0, 0))

    print("\n--- Perf : modèle (blocs CFL adaptatifs) vs solveur pas-à-pas ---")
    for N in PERF_RESOLUTIONS:
        perf_functions = test_functions.get(N) or make_test_functions(make_grid(N))
        for name in FUNCTION_NAMES:
            u0 = perf_functions[name]
            for t_phys in PHYSICAL_TIMES:
                (u_model, n_steps_model), t_model = _bench(model_rollout, u0, t_phys)
                (u_solver, _, n_dt_steps), t_solver = _bench(euler_rollout_cfl, u0, t_phys)

                rel = l2(u_model - u_solver) / (l2(u_solver) + 1e-12)
                speedup = t_solver / t_model if t_model > 0 else float('nan')
                print(f"[N={N}] [{name}] t={t_phys:g}  modèle={int(n_steps_model)} blocs ({t_model*1e3:.2f}ms)  "
                      f"solveur={int(n_dt_steps)} pas dt ({t_solver*1e3:.2f}ms)  "
                      f"err. rel. rho={float(rel[0]):.2e} m={float(rel[1]):.2e} E={float(rel[2]):.2e}  (×{speedup:.1f})")

# ------------------------------------------------------------------
# Erreur d'énergie par bande de fréquence, en fonction de l'horizon de
# rollout.
#   E_B(t)   = Σ_{k∈bande} |û(k,t)|²
#   ε_B(t)   = |E_B,pred(t) − E_B,true(t)| / (E_B,true(t) + ε)
# Les bandes sont en nombre d'onde PHYSIQUE k (le même sur toutes les
# grilles) : low k∈[1,5], mid k∈[6,15], hi1 k∈[16,31], hi2 k∈[32,47],
# hi3 k∈[48,63], hi4 k∈[64,95], hi5 k∈[96,128]. Au-delà de k=128 (modes
# qui n'existent qu'à N>256) : bande "fine", k∈[129, N/2]. Calculé sur rho.
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
                u_true = solver_rollout(u0, t_phys)
                u_pred, n_steps = model_rollout(u0, t_phys)   # même temps physique que la cible
                errs = band_error(u_pred[0], u_true[0])
                print(f"{N:>6}{name:>14}{t_phys:>8g}{int(n_steps):>9}"
                      + "".join(f"{float(errs[b]):>11.2e}" for b in FREQ_BANDS))


print_band_table("Erreur par bande de fréquence")

# ------------------------------------------------------------------
# Croissance de l'erreur ‖rho_n − rho_n(vrai)‖_L2 en fonction du temps physique
# t_n = somme des T_k : un panneau par fonction test, une courbe par résolution.
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
        u0 = test_functions[N][name]
        n_steps_err = n_steps_of(error_growth_time, u0)
        errs_pur, _, ts = error_growth(u0, n_steps_err)
        errs_pur = errs_pur[:, 0]                  # rho
        t_n = jnp.cumsum(ts)
        ax.plot(t_n, errs_pur, linewidth=1.5, label=f"N={N}")
        print(f"[{name}] N={N} ‖ε‖_L2 : t={float(t_n[0]):g} → {float(errs_pur[0]):.3e}   "
              f"t={float(t_n[-1]):g} → {float(errs_pur[-1]):.3e}")
    ax.set_title(name)
    ax.set_xlabel("t_n = somme des T_k")
    ax.set_ylabel("‖rho_n − rho_n(vrai)‖_L2")
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
import os
# autotuning XLA réactivé (niveau par défaut) : avec le niveau 0, cuDNN prend un algorithme de conv
# par défaut, potentiellement bien plus lent ; remettre la ligne ci-dessous si la compilation pose problème
# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import optax
import pickle
import math
from network_parameters import (
    MODEL, N_TRAJ, MULTIPLE_STEPS, N_TRAIN, batch_size, nb_epoch, x, SOLVER, NEW_STAGE, LAMBDA, a, P, Q,
    ONPOLICY_ENABLED, ONPOLICY_TRAJ, ONPOLICY_MAX_STEPS, ONPOLICY_DEPTHS_PER_TRAJ, ONPOLICY_REGEN_EVERY,
    EPS_ADMISSIBLE,
)
from initial_data import generate_initial_data, generate_initial_data_euler, ic_types_euler
from euler_solver import primitives, gamma
from loss import model, loss_fn, make_train_step, predict_F, V_max, normalise, T_bloc, bloc

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver
elif SOLVER == "burgers":
    from burgers_solver import burgers_solver as _solver
else:
    from euler_solver import euler_solver as _solver

print(jax.devices())
print(f"Modèle : {MODEL}")

dx = x[1] - x[0]

# noms séparés par solveur : les poids Burgers n'ont pas la même forme, et ne sont pas écrasés
CHECKPOINT_PATH = f"checkpoint_{MODEL}_{SOLVER}.pkl"
PARAMS_PATH     = f"params_{MODEL}_{SOLVER}.pkl"

key = random.PRNGKey(0)
key_init, key_train, key_val = random.split(key, 3)

# lambda = T * V / dx est fixé : chaque bloc dure T = T_bloc(u),
# calculé par état dans generate_trajectory ci-dessous.

N_VAL_TRAJ = max(1, N_TRAJ // 5)
N_VAL      = N_VAL_TRAJ * MULTIPLE_STEPS

# calcul stencil max sur tout le jeu de données pour savoir quelles valeurs prendre pour p et q
# vitesses normalisées par le V du début du bloc : dans [-1, 1] au début, peuvent dépasser à la fin
def wave_speeds(U, V):
    rho, u, p = primitives(normalise(U, V))
    c = jnp.sqrt(gamma * p / rho)
    return jnp.min(u - c), jnp.max(u + c)

def generate_trajectory(key):
    "Trajectoire de MULTIPLE_STEPS blocs en vraies valeurs, chacun integre sur T = T_bloc(u)."
    u0 = generate_initial_data_euler(key)

    def run_chunk(u, _):
        V = V_max(u)
        T = T_bloc(u)
        u_next, _, _ = _solver(u, T)
        return u_next, (u, u_next, T, wave_speeds(u_next, V))

    _, (u0s, u_finals, ts, speeds) = jax.lax.scan(run_chunk, u0, None, length=MULTIPLE_STEPS)
    return u0s, u_finals, ts, speeds


def admissible(U):
    """Etat (3, n) gardé : tout fini (NaN -> False), rho > EPS_ADMISSIBLE et p > EPS_ADMISSIBLE partout.
    Seuil absolu et non relatif à la CI : une double détente exacte peut descendre sous min(CI).
    Sous le seuil, p/rho et m/rho deviennent du bruit vis-à-vis de l'erreur du réseau : V_max peut
    exploser, ce qui fausse la normalisation et T_bloc de tout l'état et ralentit le solveur."""
    rho, _, p = primitives(U)
    return (jnp.all(jnp.isfinite(U))
            & jnp.all(rho > EPS_ADMISSIBLE)
            & jnp.all(p > EPS_ADMISSIBLE))


@jax.jit
def generate_onpolicy_pairs(key, params):

    key_ic, key_depth = random.split(key)
    keys_ic = random.split(key_ic, ONPOLICY_TRAJ)
    ics = jax.vmap(generate_initial_data_euler)(keys_ic)  # (ONPOLICY_TRAJ, 3, n)
    ic_riemann, ic_types = jax.vmap(ic_types_euler)(keys_ic)  # (ONPOLICY_TRAJ,), (ONPOLICY_TRAJ, 3)

    def rollout_model(u0):
        def body(u, _):
            u_next, _ = bloc(params, u)   # un bloc du réseau en vraies valeurs (normalisation par V incluse)
            return u_next, u_next
        _, states = jax.lax.scan(body, u0, None, length=ONPOLICY_MAX_STEPS)
        return jnp.concatenate([u0[None], states], axis=0)  # (ONPOLICY_MAX_STEPS+1, 3, n), indice 0 = IC

    all_states = jax.vmap(rollout_model)(ics)  # (ONPOLICY_TRAJ, ONPOLICY_MAX_STEPS+1, 3, n)

    # pas de principe du maximum pour Euler (les chocs créent des extrema) : on tronque la
    # trajectoire avant le 1er état non admissible (rho ou p sous EPS_ADMISSIBLE, ou NaN),
    # où le relabelling serait mal conditionné ou indéfini
    ok = jax.vmap(jax.vmap(admissible))(all_states)  # (ONPOLICY_TRAJ, MAX_STEPS+1)
    ok = jnp.concatenate([ok, jnp.zeros((ok.shape[0], 1), bool)], axis=1)
    n_valid = jnp.argmin(ok, axis=1)                                    # 1er état non admissible, >= 1 (IC ok)

    r = random.uniform(key_depth, (ONPOLICY_TRAJ, ONPOLICY_DEPTHS_PER_TRAJ))
    depths = jnp.floor(r * n_valid[:, None]).astype(jnp.int32)          # dans [0, n_valid)
    gather_traj = jax.vmap(lambda states, ds: states[ds])  # (max_steps+1,3,n),(k,) -> (k,3,n)
    u0s_corrupted = jax.lax.stop_gradient(gather_traj(all_states, depths))  # (ONPOLICY_TRAJ, k, 3, n)
    u0s_corrupted = u0s_corrupted.reshape(-1, *u0s_corrupted.shape[-2:])    # (ONPOLICY_TRAJ*k, 3, n)

    def relabel(u):
        u_true_next, _, _ = _solver(u, T_bloc(u))
        return u_true_next

    u_finals_true = jax.vmap(relabel)(u0s_corrupted)

    # diagnostic : dernier état admissible de chaque trajectoire. Pour celles qui ratent, c'est
    # l'état dont le pas suivant du réseau n'est plus admissible ; pour les saines, l'état final
    fail = n_valid <= ONPOLICY_MAX_STEPS
    last_states = jax.lax.stop_gradient(all_states[jnp.arange(ONPOLICY_TRAJ), n_valid - 1])  # (ONPOLICY_TRAJ, 3, n)

    ic_info = {"riemann": ic_riemann, "types": ic_types, "n_valid": n_valid}
    return u0s_corrupted, u_finals_true, jnp.sum(fail), jnp.mean(n_valid), last_states, fail, ic_info


@jax.jit
def relabel_batch(us):
    "Cibles du vrai solveur sur T_bloc(u) pour un lot d'états (N, 3, n)."
    return jax.vmap(lambda u: _solver(u, T_bloc(u))[0])(us)


key_train, key_traj_train, key_traj_val = random.split(key_train, 3)

u0s_traj, u_finals_traj, ts_traj, speeds_traj = jax.vmap(generate_trajectory)(random.split(key_traj_train, N_TRAJ))
u0s_training      = u0s_traj.reshape(-1, 3, x.shape[0])
u_finals_training = u_finals_traj.reshape(-1, 3, x.shape[0])
ts_training       = ts_traj.reshape(-1)   # T_bloc(u), varie selon l'état, garde juste pour verification/log

u0s_traj_val, u_finals_traj_val, ts_traj_val, speeds_traj_val = jax.vmap(generate_trajectory)(random.split(key_traj_val, N_VAL_TRAJ))
u0s_validation      = u0s_traj_val.reshape(-1, 3, x.shape[0])
u_finals_validation = u_finals_traj_val.reshape(-1, 3, x.shape[0])
ts_validation       = ts_traj_val.reshape(-1)


a_min = float(jnp.minimum(jnp.min(speeds_traj[0]), jnp.min(speeds_traj_val[0])))
a_max = float(jnp.maximum(jnp.max(speeds_traj[1]), jnp.max(speeds_traj_val[1])))
marge = max(-a_min, a_max)
print(f"Vitesses normalisées en fin de bloc : a_min={a_min:.4f}, a_max={a_max:.4f}, lambda={LAMBDA:g} "
      f"-> MARGE={marge:.4f}, {math.ceil(marge * LAMBDA)} cellules de chaque côté (réseau : {Q})")

assert u0s_training.shape[0] == N_TRAIN
n_batches     = max(1, N_TRAIN // batch_size)
n_batches_val = max(1, N_VAL // batch_size)

# Diagnostic : la durée des blocs T = T_bloc(u) varie selon l'état ; c'est T * V / dx qui est fixe.
print(f"T_batch: min={ts_training.min():.6f}, max={ts_training.max():.6f}, mean={ts_training.mean():.6f}")
print(f"dx = {dx:.6f}")
print(f"T * V / dx (facteur d'amplification, fixe) : {LAMBDA:.2f}")

# optimiseur
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-4,
    warmup_steps=5 * n_batches,
    decay_steps=nb_epoch * n_batches,
    end_value=1e-6,
)

RESUME_LR    = None   # LR fixe à la reprise ; None = continuer le cosine schedule
WEIGHT_DECAY = 1e-3

if RESUME_LR is not None and os.path.exists(CHECKPOINT_PATH):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=RESUME_LR, weight_decay=WEIGHT_DECAY)
    )
else:
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    )
train_step = make_train_step(optimizer)


@partial(jax.jit, static_argnames=("n_batches_epoch",))
def run_epoch(params, opt_state, perm, u0s_epoch, u_finals_epoch, n_batches_epoch):
    """Une epoch entière compilée en un seul scan au lieu d'une boucle Python
    dispatchant train_step batch par batch : ça évite un aller-retour
    host/device par batch (kernel-launch overhead), qui dominait le temps
    sur H100 vu la taille des batches."""
    idx_batches = perm[: n_batches_epoch * batch_size].reshape(n_batches_epoch, batch_size)

    def body(carry, idx):
        params, opt_state = carry
        params, opt_state = train_step(
            params, opt_state, u0s_epoch[idx], u_finals_epoch[idx]
        )
        return (params, opt_state), None

    (params, opt_state), _ = jax.lax.scan(
        body, (params, opt_state), idx_batches
    )
    return params, opt_state


PATIENCE = 50

IC_NOMS = ["sinus", "gaussiennes", "polynomes", "constante", "rampe",
           "marches", "creneau", "riemann", "sinus_hf", "champ_gaussien"]


def print_echecs_par_type(fail, info):
    """Taux d'échec on-policy par catégorie de CI. Hors Riemann commun, une trajectoire compte
    dans le type t si au moins une des variables (rho, u, p) est de type t."""
    riemann, types, n_valid = info["riemann"], info["types"], info["n_valid"]
    taux_global = float(jnp.mean(fail))
    print(f"           Échecs par type de CI (taux global {100 * taux_global:.1f} %, "
          f"dont {int(jnp.sum(fail & (n_valid == 1)))} dès le 1er pas) :")
    categories = [("Riemann commun", riemann)] + [
        (nom, ~riemann & jnp.any(types == t, axis=1)) for t, nom in enumerate(IC_NOMS)]
    for nom, mask in categories:
        n_cat = int(jnp.sum(mask))
        if n_cat == 0:
            continue
        n_fail = int(jnp.sum(fail & mask))
        n_fail_1 = int(jnp.sum(fail & mask & (n_valid == 1)))
        print(f"             {nom:15s} : {n_fail:5d}/{n_cat:5d} = {100 * n_fail / n_cat:5.1f} %"
              f"  (1er pas : {n_fail_1})")


def eval_losses(params, u0s, u_finals, n_batches_eval):
    "Moyenne des métriques de loss_fn (dict aux) sur n_batches_eval batches."
    totals = None
    for i in range(n_batches_eval):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        _, aux = loss_fn(params, u0s[sl], u_finals[sl])
        if totals is None:
            totals = {k: 0.0 for k in aux}
        for k, v in aux.items():
            totals[k] += v
    return {k: v / n_batches_eval for k, v in totals.items()}


if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    params        = ckpt["params"]

    try:
        opt_state = ckpt["opt_state"]
    except KeyError:
        print("⚠️  Pas d'opt_state dans le checkpoint, réinitialisation (reprise à froid).")
        opt_state = optimizer.init(params)

    start_epoch   = ckpt["epoch"] + 1
    losses_training   = ckpt["losses_training"]
    losses_validation = ckpt["losses_validation"]
    best_params   = ckpt["best_params"]

    if NEW_STAGE:
        best_val          = float("inf")
        epochs_no_improve = 0
        print(f"Reprise depuis l'epoch {start_epoch} — NOUVEAU STAGE : "
              f"best_val et epochs_no_improve réinitialisés (ancien best_val: {ckpt['best_val']:.6f})")
    else:
        best_val          = ckpt["best_val"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        print(f"Reprise depuis l'epoch {start_epoch} (meilleure val: {best_val:.6f})")
else:
    params = model.init(key_init, jnp.ones((3, x.shape[0])))  # (rho, m, E), (3, n)
    opt_state = optimizer.init(params)
    start_epoch = 0
    losses_training, losses_validation = [], []
    best_val = float("inf")
    best_params = params
    epochs_no_improve = 0
    loss0, _ = loss_fn(params, u0s_training[:batch_size], u_finals_training[:batch_size])
    print(f"[init] loss={loss0:.6f}")

if ONPOLICY_ENABLED:
    print(f"Model-in-the-loop actif : {ONPOLICY_TRAJ} trajectoires on-policy, "
          f"profondeur max {ONPOLICY_MAX_STEPS}, régénérées tous les {ONPOLICY_REGEN_EVERY} epoch(s).")

u0s_epoch, u_finals_epoch = u0s_training, u_finals_training
N_epoch = N_TRAIN
n_batches_epoch = n_batches
u0s_onpolicy = None   # dernières paires on-policy générées, pour le diagnostic

# losses toutes les 10 epochs, écrasé à chaque training (colonnes séparées par des tabulations, pour tracer)
LOSS_LOG_PATH = "losses.txt"
loss_log = open(LOSS_LOG_PATH, "w")

for epoch in range(start_epoch, nb_epoch):
    if ONPOLICY_ENABLED and epoch % ONPOLICY_REGEN_EVERY == 0:
        key_train, key_onpolicy = random.split(key_train)
        (u0s_onpolicy, u_finals_onpolicy, n_bad_onpolicy, n_valid_mean,
         last_states_onpolicy, fail_onpolicy, ic_info_onpolicy) = generate_onpolicy_pairs(key_onpolicy, params)
        n_bad_onpolicy = int(n_bad_onpolicy)
        print(f"  epoch {epoch} : {n_bad_onpolicy}/{ONPOLICY_TRAJ} trajectoires on-policy deviennent "
              f"non admissibles (rho ou p < {EPS_ADMISSIBLE:g}, ou NaN) avant {ONPOLICY_MAX_STEPS} pas (tronquées avant), "
              f"états valides moyens : {float(n_valid_mean):.1f}/{ONPOLICY_MAX_STEPS + 1}.")
        u0s_epoch      = jnp.concatenate([u0s_training, u0s_onpolicy], axis=0)
        u_finals_epoch = jnp.concatenate([u_finals_training, u_finals_onpolicy], axis=0)
        N_epoch         = u0s_epoch.shape[0]
        n_batches_epoch = max(1, N_epoch // batch_size)

    key_train, subkey = random.split(key_train)
    perm = random.permutation(subkey, N_epoch)
    params, opt_state = run_epoch(
        params, opt_state, perm, u0s_epoch, u_finals_epoch, n_batches_epoch
    )

    if epoch % 10 == 0:
        # Évaluation par batches au lieu du dataset complet
        metrics_train = eval_losses(params, u0s_training, u_finals_training, n_batches)
        metrics_val   = eval_losses(params, u0s_validation, u_finals_validation, n_batches_val)
        loss_train = metrics_train["loss"]
        loss_val   = metrics_val["loss"]

        improved = loss_val < best_val
        if improved:
            best_val = float(loss_val)
            best_params = params
            epochs_no_improve = 0
        else:
            epochs_no_improve += 10

        detail_train = " ".join(f"{k}={v:.6f}" for k, v in metrics_train.items() if k != "loss")
        detail_val   = " ".join(f"{k}={v:.6f}" for k, v in metrics_val.items() if k != "loss")
        marker = " *" if improved else ""
        print(f"Epoch {epoch} | Train: {loss_train:.6f}{' (' + detail_train + ')' if detail_train else ''}"
              f" | Val: {loss_val:.6f}{' (' + detail_val + ')' if detail_val else ''}{marker}")
        if loss_log.tell() == 0:   # en-tête à la 1re écriture : noms des métriques renvoyées par loss_fn
            loss_log.write("\t".join(["epoch"] + [f"train_{k}" for k in metrics_train] + [f"val_{k}" for k in metrics_val]) + "\n")
        loss_log.write("\t".join([str(epoch)] + [f"{float(v):.8e}" for v in [*metrics_train.values(), *metrics_val.values()]]) + "\n")
        loss_log.flush()           # lisible pendant le training

        if u0s_onpolicy is not None:
            # loss sur les états produits par le réseau lui-même : c'est là que l'on-policy doit agir
            n_batches_onpolicy = max(1, u0s_onpolicy.shape[0] // batch_size)
            metrics_onpolicy = eval_losses(params, u0s_onpolicy, u_finals_onpolicy, n_batches_onpolicy)
            detail_onpolicy = " ".join(f"{k}={v:.6f}" for k, v in metrics_onpolicy.items() if k != "loss")
            print(f"           OnPolicy: {metrics_onpolicy['loss']:.6f} ({detail_onpolicy})")

            # loss sur le dernier état admissible : trajectoires qui ratent vs trajectoires saines
            targets_last = relabel_batch(last_states_onpolicy)
            for nom, mask in (("ratées", fail_onpolicy), ("saines", ~fail_onpolicy)):
                n_mask = int(jnp.sum(mask))
                if n_mask == 0:
                    continue
                m = eval_losses(params, last_states_onpolicy[mask], targets_last[mask], max(1, n_mask // batch_size))
                detail = " ".join(f"{k}={v:.6f}" for k, v in m.items() if k != "loss")
                print(f"           Dernier état valide, {nom} ({n_mask}) : {m['loss']:.6f} ({detail})")

            print_echecs_par_type(fail_onpolicy, ic_info_onpolicy)
        losses_training.append(float(loss_train))
        losses_validation.append(float(loss_val))

        if epoch % 100 == 0:
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({
                    "epoch": epoch,
                    "params": params,
                    "opt_state": opt_state,
                    "losses_training": losses_training,
                    "losses_validation": losses_validation,
                    "best_val": best_val,
                    "best_params": best_params,
                    "epochs_no_improve": epochs_no_improve,
                }, f)
            print(f"  → Checkpoint sauvegardé (epoch {epoch})")

        if jnp.isnan(loss_train):
            print(f"NaN détecté à l'epoch {epoch}, arrêt.")
            break
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping à l'epoch {epoch} (pas d'amélioration depuis {PATIENCE} epochs). Meilleure val: {best_val:.6f}")
            break

loss_log.close()
print(f"Losses écrites dans {LOSS_LOG_PATH}")

with open(PARAMS_PATH, "wb") as f:
    pickle.dump(best_params, f)
print(f"Params sauvegardés dans {PARAMS_PATH}")
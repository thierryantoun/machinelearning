import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import jax
import jax.numpy as jnp
from jax import random
import optax
import pickle
from network_parameters import (
    MODEL, N_TRAJ, MULTIPLE_STEPS, N_TRAIN, batch_size, nb_epoch, x, SOLVER, NEW_STAGE, T_target,
    ONPOLICY_ENABLED, ONPOLICY_TRAJ, ONPOLICY_MAX_STEPS, ONPOLICY_DEPTHS_PER_TRAJ, ONPOLICY_REGEN_EVERY,
)
from initial_data import generate_initial_data
from loss import model, loss_fn, make_train_step, predict_F

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver
else:
    from burgers_solver import burgers_solver as _solver

print(jax.devices())
print(f"Modèle : {MODEL}")

dx = x[1] - x[0]

CHECKPOINT_PATH = f"checkpoint_{MODEL}.pkl"
PARAMS_PATH     = f"params_{MODEL}.pkl"

key = random.PRNGKey(0)
key_init, key_train, key_val = random.split(key, 3)

# Le solveur prend maintenant T_target (temps physique fixe) au lieu de n_steps.
# Plus de partial(_solver, n_steps=n_steps) : T_target est passe directement
# a chaque appel dans generate_trajectory ci-dessous.

N_VAL_TRAJ = max(1, N_TRAJ // 5)
N_VAL      = N_VAL_TRAJ * MULTIPLE_STEPS


def generate_trajectory(key):
    "Trajectoire de MULTIPLE_STEPS blocs, chacun integre sur le meme temps physique T_target."
    u0 = generate_initial_data(key)

    def run_chunk(u, _):
        u_next, _, t = _solver(u, T_target)
        return u_next, (u, u_next, t)

    _, (u0s, u_finals, ts) = jax.lax.scan(run_chunk, u0, None, length=MULTIPLE_STEPS)
    return u0s, u_finals, ts


def model_step(params, u):
    F = predict_F(params, u)
    return u - (T_target / dx) * (F - jnp.roll(F, 1, axis=-1))


@jax.jit
def generate_onpolicy_pairs(key, params):
    """Corrige l'exposure bias : on déroule le modèle COURANT (params) depuis
    des IC fraîches pendant ONPOLICY_MAX_STEPS pas (arrêt de gradient, ce
    n'est que de la génération de données), puis on interroge le vrai solveur
    depuis l'état atteint pour obtenir le label correct. Le réseau apprend
    ainsi à corriger ses propres états dérivés, pas seulement à reproduire des
    trajectoires toujours "propres".

    Garde-fou NaN/Inf : un modèle instable (surtout en tout début d'entraînement,
    params quasi aléatoires) peut diverger pendant le déroulé. Un seul NaN dans
    le batch casserait tous les params au gradient step suivant (le clip de
    gradient ne protège pas contre NaN, seulement contre les gradients énormes
    mais finis). Toute paire non-finie est donc remplacée par l'IC propre et
    son vrai label (garantis finis) avant d'être renvoyée.

    Les ONPOLICY_MAX_STEPS états intermédiaires de chaque trajectoire sont
    calculés de toute façon (déroulé séquentiel) : on en garde
    ONPOLICY_DEPTHS_PER_TRAJ par trajectoire (profondeurs tirées au hasard)
    au lieu d'un seul, ce qui multiplie le nombre de paires sans coût
    supplémentaire de déroulé (seul le ré-étiquetage par le vrai solveur,
    qui lui coûte réellement, scale avec ce nombre)."""
    key_ic, key_depth = random.split(key)
    ics = jax.vmap(generate_initial_data)(random.split(key_ic, ONPOLICY_TRAJ))

    def rollout_model(u0):
        def body(u, _):
            u_next = model_step(params, u)
            return u_next, u_next
        _, states = jax.lax.scan(body, u0, None, length=ONPOLICY_MAX_STEPS)
        return states  # (ONPOLICY_MAX_STEPS, n) : état après 1..ONPOLICY_MAX_STEPS pas modèle

    all_states = jax.vmap(rollout_model)(ics)  # (ONPOLICY_TRAJ, ONPOLICY_MAX_STEPS, n)
    depths = random.randint(key_depth, (ONPOLICY_TRAJ, ONPOLICY_DEPTHS_PER_TRAJ), 0, ONPOLICY_MAX_STEPS)
    gather_traj = jax.vmap(lambda states, ds: states[ds])  # (max_steps,n),(k,) -> (k,n)
    u0s_corrupted = jax.lax.stop_gradient(gather_traj(all_states, depths))  # (ONPOLICY_TRAJ, k, n)
    u0s_corrupted = u0s_corrupted.reshape(-1, u0s_corrupted.shape[-1])      # (ONPOLICY_TRAJ*k, n)

    def relabel(u):
        u_true_next, _, _ = _solver(u, T_target)
        return u_true_next

    u_finals_true = jax.vmap(relabel)(u0s_corrupted)

    u_finals_ics_per_traj = jax.vmap(relabel)(ics)                                    # (ONPOLICY_TRAJ, n)
    ics_flat          = jnp.repeat(ics, ONPOLICY_DEPTHS_PER_TRAJ, axis=0)              # (ONPOLICY_TRAJ*k, n)
    u_finals_ics_flat = jnp.repeat(u_finals_ics_per_traj, ONPOLICY_DEPTHS_PER_TRAJ, axis=0)

    is_bad = (jnp.any(~jnp.isfinite(u0s_corrupted), axis=-1)
              | jnp.any(~jnp.isfinite(u_finals_true), axis=-1))
    u0s_corrupted = jnp.where(is_bad[:, None], ics_flat, u0s_corrupted)
    u_finals_true = jnp.where(is_bad[:, None], u_finals_ics_flat, u_finals_true)

    return u0s_corrupted, u_finals_true, jnp.sum(is_bad)


key_train, key_traj_train, key_traj_val = random.split(key_train, 3)

u0s_traj, u_finals_traj, ts_traj = jax.vmap(generate_trajectory)(random.split(key_traj_train, N_TRAJ))
u0s_training      = u0s_traj.reshape(-1, x.shape[0])
u_finals_training = u_finals_traj.reshape(-1, x.shape[0])
ts_training       = ts_traj.reshape(-1)   # constant = T_target, garde juste pour verification/log

u0s_traj_val, u_finals_traj_val, ts_traj_val = jax.vmap(generate_trajectory)(random.split(key_traj_val, N_VAL_TRAJ))
u0s_validation      = u0s_traj_val.reshape(-1, x.shape[0])
u_finals_validation = u_finals_traj_val.reshape(-1, x.shape[0])
ts_validation       = ts_traj_val.reshape(-1)


assert u0s_training.shape[0] == N_TRAIN
n_batches     = max(1, N_TRAIN // batch_size)
n_batches_val = max(1, N_VAL // batch_size)

# Diagnostic : ts_training doit maintenant etre constant (= T_target partout).
# On garde le print par securite, pour detecter tout ecart inattendu (bug de
# generation, T_target mal propage, etc.) plutot que de le supprimer purement.
print(f"T_batch: min={ts_training.min():.6f}, max={ts_training.max():.6f}, mean={ts_training.mean():.6f} "
      f"(devrait etre constant = T_target = {T_target})")
print(f"dx = {dx:.6f}")
print(f"T_target/dx (facteur d'amplification, fixe) : {T_target/dx:.2f}")

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

PATIENCE = 50


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
    params = model.init(key_init, jnp.ones(x.shape[0]))
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

for epoch in range(start_epoch, nb_epoch):
    if ONPOLICY_ENABLED and epoch % ONPOLICY_REGEN_EVERY == 0:
        key_train, key_onpolicy = random.split(key_train)
        u0s_onpolicy, u_finals_onpolicy, n_bad_onpolicy = generate_onpolicy_pairs(key_onpolicy, params)
        n_bad_onpolicy = int(n_bad_onpolicy)
        if n_bad_onpolicy > 0:
            print(f"  ⚠️  epoch {epoch} : {n_bad_onpolicy}/{u0s_onpolicy.shape[0]} paires on-policy "
                  f"non-finies (modèle divergent), remplacées par IC propre.")
        u0s_epoch      = jnp.concatenate([u0s_training, u0s_onpolicy], axis=0)
        u_finals_epoch = jnp.concatenate([u_finals_training, u_finals_onpolicy], axis=0)
        N_epoch         = u0s_epoch.shape[0]
        n_batches_epoch = max(1, N_epoch // batch_size)

    key_train, subkey = random.split(key_train)
    perm = random.permutation(subkey, N_epoch)
    for i in range(n_batches_epoch):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        params, opt_state = train_step(
            params, opt_state,
            u0s_epoch[idx], u_finals_epoch[idx]
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

with open(PARAMS_PATH, "wb") as f:
    pickle.dump(best_params, f)
print(f"Params sauvegardés dans {PARAMS_PATH}")
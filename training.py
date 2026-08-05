import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import jax
import jax.numpy as jnp
from jax import random
import optax
import pickle
from functools import partial
from network_parameters import MODEL, N_TRAJ, MULTIPLE_STEPS, N_TRAIN, batch_size, nb_epoch, x, SOLVER, NEW_STAGE
from initial_data import generate_initial_data
from loss import model, loss_fn, make_train_step


if SOLVER == "advection":
    from advection_solver import advection_solver as _solver, n_steps
else:
    from burgers_solver import burgers_solver as _solver, n_steps

print(jax.devices())
print(f"Modèle : {MODEL}")

CHECKPOINT_PATH = f"checkpoint_{MODEL}.pkl"
PARAMS_PATH     = f"params_{MODEL}.pkl"

key = random.PRNGKey(0)
key_init, key_train, key_val = random.split(key, 3)

_solver = partial(_solver, n_steps=n_steps)

N_VAL_TRAJ = max(1, N_TRAJ // 5)
N_VAL      = N_VAL_TRAJ * MULTIPLE_STEPS

def generate_trajectory(key):
    "Trajectoire de n_steps*MULTIPLE_STEPS pas, coupée en paires (u_{k*n_steps}, u_{(k+1)*n_steps})."
    u0 = generate_initial_data(key)

    def run_chunk(u, _):
        u_next, _, t = _solver(u)
        return u_next, (u, u_next, t)

    _, (u0s, u_finals, ts) = jax.lax.scan(run_chunk, u0, None, length=MULTIPLE_STEPS)
    return u0s, u_finals, ts

key_train, key_traj_train, key_traj_val = random.split(key_train, 3)

u0s_traj, u_finals_traj, ts_traj = jax.vmap(generate_trajectory)(random.split(key_traj_train, N_TRAJ))
u0s_training      = u0s_traj.reshape(-1, x.shape[0])
u_finals_training = u_finals_traj.reshape(-1, x.shape[0])
ts_training       = ts_traj.reshape(-1)

u0s_traj_val, u_finals_traj_val, ts_traj_val = jax.vmap(generate_trajectory)(random.split(key_traj_val, N_VAL_TRAJ))
u0s_validation      = u0s_traj_val.reshape(-1, x.shape[0])
u_finals_validation = u_finals_traj_val.reshape(-1, x.shape[0])
ts_validation       = ts_traj_val.reshape(-1)

assert u0s_training.shape[0] == N_TRAIN
n_batches     = max(1, N_TRAIN // batch_size)
n_batches_val = max(1, N_VAL // batch_size)

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


def eval_losses(params, u0s, u_finals, ts, n_batches_eval):
    "Moyenne des métriques de loss_fn (dict aux) sur n_batches_eval batches."
    totals = None
    for i in range(n_batches_eval):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        _, aux = loss_fn(params, u0s[sl], u_finals[sl], ts[sl])
        if totals is None:
            totals = {k: 0.0 for k in aux}
        for k, v in aux.items():
            totals[k] += v
    return {k: v / n_batches_eval for k, v in totals.items()}


if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    params        = ckpt["params"]
    opt_state     = optimizer.init(params)
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
    loss0, _ = loss_fn(params, u0s_training[:batch_size], u_finals_training[:batch_size], ts_training[:batch_size])
    print(f"[init] loss={loss0:.6f}")

for epoch in range(start_epoch, nb_epoch):
    key_train, subkey = random.split(key_train)
    perm = random.permutation(subkey, N_TRAIN)
    for i in range(n_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        params, opt_state = train_step(
            params, opt_state,
            u0s_training[idx], u_finals_training[idx], ts_training[idx]
        )

    if epoch % 10 == 0:
        # Évaluation par batches au lieu du dataset complet
        metrics_train = eval_losses(params, u0s_training, u_finals_training, ts_training, n_batches)
        metrics_val   = eval_losses(params, u0s_validation, u_finals_validation, ts_validation, n_batches_val)
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
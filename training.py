import jax
import jax.numpy as jnp
from jax import random
import optax
import pickle
from functools import partial
from network_parameters import N, batch_size, nb_epoch, x, SOLVER
from initial_data import generate_initial_data
from loss import model, loss_fn, make_train_step

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver, n_steps
else:
    from burgers_solver import burgers_solver as _solver, n_steps

CHECKPOINT_PATH = "checkpoint.pkl"

key = random.PRNGKey(0)
key_init, key_train, key_val = random.split(key, 3)

_solver = partial(_solver, n_steps=n_steps)


N_TRAJ   = 50 # nombre de trajectoires
N_CHUNKS = 100 # nombre de paires (u0, F) par trajectoire


def generate_trajectory(key):
    "Une trajectoire de N_CHUNKS * n_steps pas."
    u0 = generate_initial_data(key)

    def run_chunk(u, _):
        u_final, _, t = _solver(u)
        return u_final, (u, u_final, t)

    _, (u0s, u_finals, ts) = jax.lax.scan(run_chunk, u0, None, length=N_CHUNKS)
    return u0s, u_finals, ts


N_total_paires = N_TRAJ * N_CHUNKS

u0s_training, u_finals_training, ts_training = jax.vmap(generate_trajectory)(random.split(key_train, N_TRAJ))
u0s_training    = u0s_training.reshape(N_total_paires, -1)
u_finals_training = u_finals_training.reshape(N_total_paires, -1)
ts_training     = ts_training.reshape(N_total_paires)

u0s_validation, u_finals_validation, ts_validation = jax.vmap(generate_trajectory)(random.split(key_val, N_TRAJ))
u0s_validation    = u0s_validation.reshape(N_total_paires, -1)
u_finals_validation = u_finals_validation.reshape(N_total_paires, -1)
ts_validation     = ts_validation.reshape(N_total_paires)

n_batches = N_total_paires // batch_size

# optimiseur
schedule = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=nb_epoch * n_batches)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule),
)
train_step = make_train_step(optimizer)

PATIENCE = 50

import os
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    params        = ckpt["params"]
    opt_state     = ckpt["opt_state"]
    start_epoch   = ckpt["epoch"] + 1
    losses_training   = ckpt["losses_training"]
    losses_validation = ckpt["losses_validation"]
    best_val      = ckpt["best_val"]
    best_params   = ckpt["best_params"]
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
    for i in range(n_batches):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        params, opt_state = train_step(
            params, opt_state,
            u0s_training[sl], u_finals_training[sl], ts_training[sl]
        )

    if epoch % 10 == 0:
        loss_train, _ = loss_fn(params, u0s_training, u_finals_training, ts_training)
        loss_val,   _ = loss_fn(params, u0s_validation, u_finals_validation, ts_validation)

        improved = loss_val < best_val
        if improved:
            best_val = float(loss_val)
            best_params = params
            epochs_no_improve = 0
        else:
            epochs_no_improve += 10

        marker = " *" if improved else ""
        print(f"Epoch {epoch} | Train: {loss_train:.6f} | Val: {loss_val:.6f}{marker}")
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

with open("params.pkl", "wb") as f:
    pickle.dump(best_params, f)
print("Params sauvegardés dans params.pkl")

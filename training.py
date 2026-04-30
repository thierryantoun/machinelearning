import jax
import jax.numpy as jnp
from jax import random
import optax

from functools import partial
from network_parameters import N, K, batch_size, nb_epoch, n_batches, x
from initial_data import generate_initial_data
from advection_solver import advection_solver, n_steps
from loss import model, loss_fn, make_train_step

lambda_consistency = 1

key = random.PRNGKey(0)
key_train, key_val = random.split(key)
x_dummy = jnp.ones((64, 1))
params = model.init(key, x_dummy)

# génération des données
keys_train = random.split(key_train, N)
u0s_training = jax.vmap(generate_initial_data)(keys_train)
_solver = partial(advection_solver, n_steps=n_steps)
F_targets_training, _ = jax.vmap(_solver)(u0s_training)
u0s_training      = u0s_training[:, :, None]       # (N, n, 1)
F_targets_training = F_targets_training[:, :, None] # (N, n, 1)

keys_val = random.split(key_val, N)
u0s_validation = jax.vmap(generate_initial_data)(keys_val)
F_targets_validation, _ = jax.vmap(_solver)(u0s_validation)
u0s_validation      = u0s_validation[:, :, None]
F_targets_validation = F_targets_validation[:, :, None]

# setup de l'optimiseur et du train step
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4),
)
opt_state = optimizer.init(params)
train_step = make_train_step(optimizer, lambda_consistency)

# vérification des pertes initiales (avant tout entraînement)
loss0, aux0 = loss_fn(params, u0s_training[:batch_size], F_targets_training[:batch_size], lambda_consistency)
print(f"[init] total={loss0:.6f}, main={aux0['loss_main']:.6f}, consistency={aux0['loss_consistency']:.6f}")

losses_training = []
losses_validation = []
for epoch in range(nb_epoch):
    for i in range(n_batches):
        u0s_batch = u0s_training[i*batch_size:(i+1)*batch_size]
        F_batch   = F_targets_training[i*batch_size:(i+1)*batch_size]
        params, opt_state = train_step(params, opt_state, u0s_batch, F_batch)

    if epoch % 10 == 0:
        loss_train, aux_train = loss_fn(params, u0s_training, F_targets_training, lambda_consistency)
        loss_val,   aux_val   = loss_fn(params, u0s_validation, F_targets_validation, lambda_consistency)
        print(
            f"Epoch {epoch} | "
            f"Train — total: {loss_train:.6f}, main: {aux_train['loss_main']:.6f}, consistency: {aux_train['loss_consistency']:.6f} | "
            f"Val   — total: {loss_val:.6f},   main: {aux_val['loss_main']:.6f},   consistency: {aux_val['loss_consistency']:.6f}"
        )
        losses_training.append(float(loss_train))
        losses_validation.append(float(loss_val))
        if jnp.isnan(loss_train):
            print(f"NaN détecté à l'epoch {epoch}, arrêt de l'entraînement.")
            break

import pickle
with open("params.pkl", "wb") as f:
    pickle.dump(params, f)
print("Params sauvegardés dans params.pkl")

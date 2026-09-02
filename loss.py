import jax
import jax.numpy as jnp
import optax
from network import model
from network_parameters import x, a, cfl, SOLVER, MODEL, n, lambda_hf, K, T_target

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver
else:
    from burgers_solver import burgers_solver as _solver

dx = x[1] - x[0]

def predict_F(params, u0):
    mean_u0 = jnp.mean(u0)
    N      = model.apply(params, u0)
    N_mean = model.apply(params, jnp.full_like(u0, mean_u0))
    if SOLVER == "advection":
        consistance = a * mean_u0
    else:
        consistance = mean_u0**2 / 2.0
    return consistance + N - N_mean


kappa = 0.25
Ktot  = n // 2 + 1
k0    = int(kappa * Ktot)

k_max_fno = K
k1_fno    = k_max_fno
k1_lgno   = Ktot
 
 
def _loss_fno(params, u0s_batch, u_finals_batch):
    F_pred = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    u_pred = u0s_batch - (T_target / dx) * (F_pred - jnp.roll(F_pred, 1, axis=-1))

    erreur_vec = u_pred - u_finals_batch
    erreur = jnp.sqrt(jnp.sum(erreur_vec ** 2, axis=1) + 1e-12)
    norme  = jnp.sqrt(jnp.sum(u_finals_batch ** 2, axis=1) + 1e-12)
    loss_phys = jnp.mean(erreur / norme)
 
    e_hat   = jnp.fft.rfft(erreur_vec, axis=-1)
    e_hf    = e_hat[:, k0:k1_fno]
    loss_hf = 1 / (k1_fno - k0) * jnp.mean(jnp.sum(jnp.abs(e_hf) ** 2, axis=-1))
 
    loss = loss_phys + lambda_hf * loss_hf
    return loss, {"loss": loss, "loss_phys": loss_phys, "loss_hf": loss_hf}
 
 
def _loss_lgno(params, u0s_batch, u_finals_batch):
    F_pred = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    u_pred = u0s_batch - (T_target / dx) * (F_pred - jnp.roll(F_pred, 1, axis=-1))
    erreur = u_pred - u_finals_batch
    loss_phys = jnp.mean(jnp.sum(jnp.abs(erreur), axis=1) / n)
 
    e_hat   = jnp.fft.rfft(erreur, axis=-1)
    e_hf    = e_hat[:, k0:k1_lgno]
    loss_hf = 1 / (k1_lgno - k0) * jnp.mean(jnp.sum(jnp.abs(e_hf) ** 2, axis=-1))
 
    loss = loss_phys + lambda_hf * loss_hf
    return loss, {"loss": loss, "loss_phys": loss_phys, "loss_hf": loss_hf}



loss_fn = jax.jit(_loss_fno if MODEL == "fno" else _loss_lgno)


def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, u_finals_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, u_finals_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step
import jax
import jax.numpy as jnp
import optax
from network import model, FNO1D
from network_parameters import x, a, cfl, SOLVER, MODEL, n, lambda_hf, lambda_phys, K, T_target, P, Q

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver
else:
    from burgers_solver import burgers_solver as _solver

dx = x[1] - x[0]

# F_{j+1/2} = f(wb_j) + N[w_j] - N[wb_j], N[wb_j] évalué sur la fenêtre constante (wb_j, ..., wb_j)
# fenêtre j constante -> w_j = w_bar_j -> F = f(wb_j), quels que soient les poids
def predict_F(params, u0):
    S     = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.pad(u0, (P, Q), mode="wrap"))])  # (n+P+Q+1,)
    wb    = (S[P + Q + 1:] - S[:u0.shape[-1]]) / (P + Q + 1)     # moyenne sur chaque stencil sans construire la grosse matrice
    N     = model.apply(params, u0)                   # (n,) convolution sur u0
    N_bar = model.apply(params, wb, method=FNO1D.constant)  # (n,) fenêtres constantes (wb_j, ..., wb_j)
    if SOLVER == "advection":
        consistance = a * wb
    else:
        consistance = wb**2 / 2.0
    return consistance + N - N_bar


Ktot  = n // 2 + 1
kappa     = 0.8
k0        = int(kappa * Ktot)


def _loss_fno(params, u0s_batch, u_finals_batch):
    F_pred = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    u_pred = u0s_batch - (T_target / dx) * (F_pred - jnp.roll(F_pred, 1, axis=-1))

    erreur_vec = u_pred - u_finals_batch
    erreur = jnp.sqrt(jnp.sum(erreur_vec ** 2, axis=1) + 1e-12)
    norme  = jnp.sqrt(jnp.sum(u_finals_batch ** 2, axis=1) + 1e-12)
    loss_phys = jnp.mean(erreur / norme)

    e_hat   = jnp.fft.rfft(erreur_vec, axis=-1)
    e_hf    = e_hat[:, k0:Ktot]
    loss_hf = 1 / (Ktot- k0) * jnp.mean(jnp.sum(jnp.abs(e_hf), axis=-1))

    loss = lambda_phys * loss_phys + lambda_hf * loss_hf
    return loss, {"loss": loss, "loss_phys": loss_phys, "loss_hf": loss_hf}

loss_fn = jax.jit(_loss_fno)

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, u_finals_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, u_finals_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step
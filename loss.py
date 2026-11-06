import jax
import jax.numpy as jnp
import optax
from network import model, stencil_conv1D
from network_parameters import a, cfl, SOLVER, MODEL, n, lambda_hf, lambda_phys, LAMBDA, P, Q

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver
elif SOLVER == "burgers":
    from burgers_solver import burgers_solver as _solver
else: 
    from euler_solver import euler_solver as _solver, flux, primitives, gamma


# V = max_j (|u_j| + c_j), un nombre par état, recalculé à chaque bloc
def V_max(U):
    rho, u, p = primitives(U)
    return jnp.max(jnp.abs(u) + jnp.sqrt(gamma * p / rho))


# F_{j+1/2} = flux(wb_L_j, wb_R_j) + N[w_j] - N[wb_j], N[wb_j] évalué sur la fenêtre constante (wb_j, ..., wb_j)
# fenêtre j constante -> w_j = wb_L_j = wb_R_j = wb_j -> F = f(wb_j), quels que soient les poids
def predict_F(params, U0):
    nx    = U0.shape[-1]
    S     = jnp.concatenate([jnp.zeros((3, 1)), jnp.cumsum(jnp.pad(U0, ((0, 0), (P, Q)), mode="wrap"), axis=-1)], axis=-1)  # (3, n+P+Q+1)
    wb    = (S[:, P + Q + 1:] - S[:, :nx]) / (P + Q + 1)     # moyenne sur chaque stencil sans construire la grosse matrice
    wb_L  = (S[:, P + 1:nx + P + 1] - S[:, :nx]) / (P + 1)   # moyenne de U_{j-P} .. U_j, à gauche de la face j+1/2
    wb_R  = (S[:, P + Q + 1:] - S[:, P + 1:nx + P + 1]) / Q  # moyenne de U_{j+1} .. U_{j+Q}, à droite de la face j+1/2
    N     = model.apply(params, U0)                   # (3, n) convolution sur U0
    N_bar = model.apply(params, wb, method=stencil_conv1D.constant)  # (3, n) fenêtres constantes (wb_j, ..., wb_j)
    if SOLVER == "advection":
        consistance = a * wb
    elif SOLVER == "burgers":
        consistance = wb**2 / 2
    else:
        consistance = flux(wb_L, wb_R)
    return consistance + N - N_bar


# durée d'un bloc : T = LAMBDA * dx / V, quelle que soit la grille de U
def T_bloc(U):
    return LAMBDA / U.shape[-1] / V_max(U)


# un bloc du réseau : T / dx = LAMBDA / V
def bloc(params, U):
    F = predict_F(params, U)
    return U - LAMBDA / V_max(U) * (F - jnp.roll(F, 1, axis=-1)), T_bloc(U)


# blocs du réseau tant qu'on ne dépasse pas T_final ; le réseau est entraîné à lambda fixe,
# donc on ne raccourcit pas le dernier bloc : le solveur fait le reste (moins d'un bloc)
# renvoie (U(T_final), nombre de blocs du réseau)
@jax.jit
def rollout(params, U0, T_final):
    def cond_fn(carry):
        U, t, n_blocks = carry
        return t + T_bloc(U) <= T_final

    def body_fn(carry):
        U, t, n_blocks = carry
        U_new, T_k = bloc(params, U)
        return (U_new, t + T_k, n_blocks + 1)

    U, t, n_blocks = jax.lax.while_loop(cond_fn, body_fn, (U0, 0.0, 0))
    return _solver(U, T_final - t)[0], n_blocks


Ktot  = n // 2 + 1
kappa     = 0.8
k0        = int(kappa * Ktot)


# u0s_batch, u_finals_batch : (batch, 3, n), vraies valeurs, u_final à T_bloc(u0) de u0
def _loss_stencil_conv1D(params, u0s_batch, u_finals_batch):
    Vs = jax.vmap(V_max)(u0s_batch)[:, None, None]             # (batch, 1, 1)
    F_pred = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    u_pred = u0s_batch - LAMBDA / Vs * (F_pred - jnp.roll(F_pred, 1, axis=-1))

    erreur_vec = u_pred - u_finals_batch
    erreur = jnp.sqrt(jnp.sum(erreur_vec ** 2, axis=-1) + 1e-12)  # (batch, 3)
    # norme moyenne sur le batch, par variable : E ne domine pas rho, et rho*u peut être nul sur une trajectoire
    norme  = jnp.mean(jnp.sqrt(jnp.sum(u_finals_batch ** 2, axis=-1) + 1e-12), axis=0, keepdims=True)  # (1, 3)
    loss_phys = jnp.mean(erreur / norme)

    e_hat   = jnp.fft.rfft(erreur_vec, axis=-1)
    e_hf    = e_hat[..., k0:Ktot]
    loss_hf = 1 / (Ktot- k0) * jnp.mean(jnp.sum(jnp.abs(e_hf), axis=-1) / norme)

    loss = lambda_phys * loss_phys + lambda_hf * loss_hf
    return loss, {"loss": loss, "loss_phys": loss_phys, "loss_hf": loss_hf}

loss_fn = jax.jit(_loss_stencil_conv1D)

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, u_finals_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, u_finals_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step
import jax
import jax.numpy as jnp
import optax
from network import model, stencil_conv1D
from network_parameters import a, cfl, SOLVER, MODEL, n, lambda_hf, lambda_phys, lambda_pos, EPS_POS, PHYS_NORM, LAMBDA, P, Q

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

# (rho, m, E) -> (rho, m / V, E / V²) : u -> u / V, p -> p / V², et max(|u| + c) = 1 si V = V_max(U)
def normalise(U, V):
    return jnp.stack([U[0], U[1] / V, U[2] / V**2])

def denormalise(U, V):
    return jnp.stack([U[0], U[1] * V, U[2] * V**2])


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


# un bloc du réseau sur les vraies valeurs
def bloc(params, U):
    V   = V_max(U)
    U_t = normalise(U, V)
    F_t = predict_F(params, U_t)
    U_t = U_t - LAMBDA * (F_t - jnp.roll(F_t, 1, axis=-1))
    return denormalise(U_t, V), T_bloc(U)


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


# Positivité : terme à seuil, nul tant que rho_pred >= EPS_POS * rho_cible et q_pred >= EPS_POS * q_cible,
# avec q = rho * E - m^2 / 2 = rho * p / (gamma - 1) : si rho > 0, p > 0 <=> q > 0, et q est polynomial
# (pas de division par un rho_pred proche de 0, gradient borné). Divisé par la moyenne de la cible ;
# somme sur les cellules pour qu'une seule cellule fautive pèse. Invariant par la normalisation par V.
def loss_positivite(u_pred, u_cible):  # (batch, 3, n) chacun
    def rho_q(U):
        return U[:, 0], U[:, 0] * U[:, 2] - 0.5 * U[:, 1]**2
    rho_pred, q_pred = rho_q(u_pred)
    rho_cib,  q_cib  = rho_q(u_cible)
    viol_rho = jax.nn.relu(EPS_POS * rho_cib - rho_pred) / jnp.mean(rho_cib, axis=-1, keepdims=True)
    viol_q   = jax.nn.relu(EPS_POS * q_cib - q_pred)     / jnp.mean(q_cib, axis=-1, keepdims=True)
    return jnp.mean(jnp.sum(viol_rho + viol_q, axis=-1))


# u0s_batch, u_finals_batch : (batch, 3, n), vraies valeurs, u_final à T_bloc(u0) de u0
def _loss_stencil_conv1D(params, u0s_batch, u_finals_batch):
    Vs = jax.vmap(V_max)(u0s_batch)
    u0s_batch      = jax.vmap(normalise)(u0s_batch, Vs)        # entrée et cible avec le même V, celui de l'entrée
    u_finals_batch = jax.vmap(normalise)(u_finals_batch, Vs)
    F_pred = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    u_pred = u0s_batch - LAMBDA * (F_pred - jnp.roll(F_pred, 1, axis=-1))

    erreur_vec = u_pred - u_finals_batch
    # norme L2 moyenne sur le batch, par variable : E ne domine pas rho, et rho*u peut être nul sur une trajectoire.
    # Sert aussi à loss_hf, quelle que soit PHYS_NORM (sinon l'échelle de loss_hf change d'un facteur ~sqrt(n))
    norme  = jnp.mean(jnp.sqrt(jnp.sum(u_finals_batch ** 2, axis=-1) + 1e-12), axis=0, keepdims=True)  # (1, 3)
    if PHYS_NORM == "L1":
        # L1 relative : régression vers la médiane, un choc décalé d'une cellule pèse moins -> moins d'étalement
        erreur   = jnp.sum(jnp.abs(erreur_vec), axis=-1)                                              # (batch, 3)
        norme_l1 = jnp.mean(jnp.sum(jnp.abs(u_finals_batch), axis=-1), axis=0, keepdims=True) + 1e-12  # (1, 3)
        loss_phys = jnp.mean(erreur / norme_l1)
    else:
        erreur = jnp.sqrt(jnp.sum(erreur_vec ** 2, axis=-1) + 1e-12)  # (batch, 3)
        loss_phys = jnp.mean(erreur / norme)

    e_hat   = jnp.fft.rfft(erreur_vec, axis=-1)
    e_hf    = e_hat[..., k0:Ktot]
    loss_hf = 1 / (Ktot- k0) * jnp.mean(jnp.sum(jnp.abs(e_hf), axis=-1) / norme)

    loss_pos = loss_positivite(u_pred, u_finals_batch)

    loss = lambda_phys * loss_phys + lambda_hf * loss_hf + lambda_pos * loss_pos
    return loss, {"loss": loss, "loss_phys": loss_phys, "loss_hf": loss_hf, "loss_pos": loss_pos}

loss_fn = jax.jit(_loss_stencil_conv1D)

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, u_finals_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, u_finals_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step
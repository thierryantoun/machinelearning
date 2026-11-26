import jax
import jax.numpy as jnp
import optax
from network import model, stencil_conv1D
from network_parameters import a, cfl, SOLVER, MODEL, n, lambda_hf, lambda_phys, lambda_pos, EPS_POS, PHYS_NORM, LAMBDA, P, Q, MARGE

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

# R = max_j rho_j, échelle de densité, un nombre par état
def R_max(U):
    return jnp.max(U[0])

# adimensionnement classique d'Euler (3 échelles : L = 1, rho_ref = R, u_ref = V ; p_ref = E_ref = R V²)
# (rho, m, E) -> (rho / R, m / (R V), E / (R V²)) : rho -> rho / R, u -> u / V, p -> p / (R V²)
# l'état normalisé reste un état d'Euler (même gamma), max(|u| + c) = 1 si V = V_max(U), max(rho) = 1 si R = R_max(U)
def normalise(U, V, R):
    return jnp.stack([U[0] / R, U[1] / (R * V), U[2] / (R * V**2)])

def denormalise(U, V, R):
    return jnp.stack([U[0] * R, U[1] * (R * V), U[2] * (R * V**2)])


# Échelles par stencil. Fenêtre maximale de la face j+1/2 : cellules j-P .. j+Q (k = -P .. Q), indices périodiques (n, P+Q+1)
K_FEN = jnp.arange(-P, Q + 1)

def indices_fenetres(nx):
    return (jnp.arange(nx)[:, None] + K_FEN[None, :]) % nx

# |u| + c par cellule, (n,)
def vitesses(U):
    rho, u, p = primitives(U)
    return jnp.abs(u) + jnp.sqrt(gamma * p / rho)

# cône local de la face j+1/2 : lambda_loc = LAMBDA * V_fen / V_max(U) cellules parcourues par l'onde locale la plus rapide
# pendant le bloc (V_fen : max de |u| + c sur la fenêtre maximale) -> q_loc = ceil(MARGE * lambda_loc) dans [1, Q],
# p_loc = q_loc - 1 ; masque (n, P+Q+1) vrai sur k = -p_loc .. q_loc. lambda_loc < 1 -> 2 cellules (j, j+1)
def masque_cone(U):
    idx   = indices_fenetres(U.shape[-1])
    a     = vitesses(U)
    V_fen = jnp.max(a[idx], axis=-1)                                   # (n,)
    lambda_loc = LAMBDA * V_fen / jnp.max(a)
    q_loc = jnp.clip(jnp.ceil(MARGE * lambda_loc), 1, Q).astype(jnp.int32)
    return (K_FEN[None, :] >= 1 - q_loc[:, None]) & (K_FEN[None, :] <= q_loc[:, None])

# V_loc = max de |u| + c et R_loc = max de rho sur le cône local de chaque face, (n,) chacun
def V_R_loc(U, masque):
    idx = indices_fenetres(U.shape[-1])
    V_loc = jnp.max(jnp.where(masque, vitesses(U)[idx], 0.0), axis=-1)
    R_loc = jnp.max(jnp.where(masque, U[0][idx], 0.0), axis=-1)
    return V_loc, R_loc


# F_{j+1/2} = flux(wb_L_j, wb_R_j) + N[w_j] - N[wb_j], N[wb_j] évalué sur la fenêtre constante (wb_j, ..., wb_j)
# fenêtre j constante -> w_j = wb_L_j = wb_R_j = wb_j -> F = f(wb_j), quels que soient les poids
def predict_F(params, U0):
    if SOLVER == "euler":
        return predict_F_local(params, U0)
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


# Euler, stencil local par face j+1/2 : cône k = -p_loc .. q_loc (masque_cone), normalisé par (V_loc, R_loc) du cône.
# F = flux(wb_L, wb_R) + (R V, R V², R V³)_loc * (N[w_loc] - N[wb_loc]), moyennes wb, wb_L, wb_R sur le cône seul.
# U0 et F dans les unités de U0 (normalisation globale du bloc) ; fenêtre constante sur le cône -> N = N_bar -> F = f(wb)
def predict_F_local(params, U0):
    idx    = indices_fenetres(U0.shape[-1])
    masque = masque_cone(U0)                                 # (n, P+Q+1)
    V, R   = V_R_loc(U0, masque)                             # (n,) chacun
    ech    = jnp.stack([R, R * V, R * V**2])                 # (3, n) échelles de (rho, m, E) par face
    W      = U0[:, idx]                                      # (3, n, P+Q+1) fenêtres maximales
    M      = masque.astype(U0.dtype)
    M_L    = M * (K_FEN <= 0)                                # j-p_loc .. j, à gauche de la face
    M_R    = M * (K_FEN >= 1)                                # j+1 .. j+q_loc, à droite
    wb     = jnp.sum(W * M,   axis=-1) / jnp.sum(M,   axis=-1)
    wb_L   = jnp.sum(W * M_L, axis=-1) / jnp.sum(M_L, axis=-1)
    wb_R   = jnp.sum(W * M_R, axis=-1) / jnp.sum(M_R, axis=-1)
    W_loc  = W / ech[:, :, None] * M                         # normalisées par face, zéros hors du cône
    N      = model.apply(params, jnp.transpose(W_loc, (1, 2, 0)), method=stencil_conv1D.fenetres)
    N_bar  = model.apply(params, wb / ech, masque, method=stencil_conv1D.constant_masque)
    ech_F  = jnp.stack([R * V, R * V**2, R * V**3])          # échelles des flux de masse, quantité de mouvement, énergie
    return flux(wb_L, wb_R) + ech_F * (N - N_bar)


# durée d'un bloc : T = LAMBDA * dx / V, quelle que soit la grille de U
def T_bloc(U):
    return LAMBDA / U.shape[-1] / V_max(U)


# un bloc du réseau sur les vraies valeurs
def bloc(params, U):
    V   = V_max(U)
    R   = R_max(U)
    U_t = normalise(U, V, R)
    F_t = predict_F(params, U_t)
    U_t = U_t - LAMBDA * (F_t - jnp.roll(F_t, 1, axis=-1))
    return denormalise(U_t, V, R), T_bloc(U)


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
# (pas de division par un rho_pred proche de 0). rho divisé par la moyenne de la cible ; q divisé par
# q_cible de la cellule : une violation pèse autant dans une zone à basse pression (blast : q ~ 1e-5 x moyenne)
# qu'ailleurs. Somme sur les cellules pour qu'une seule cellule fautive pèse. Invariant par la normalisation par (V, R).
def loss_positivite(u_pred, u_cible):  # (batch, 3, n) chacun
    def rho_q(U):
        return U[:, 0], U[:, 0] * U[:, 2] - 0.5 * U[:, 1]**2
    rho_pred, q_pred = rho_q(u_pred)
    rho_cib,  q_cib  = rho_q(u_cible)
    viol_rho = jax.nn.relu(EPS_POS * rho_cib - rho_pred) / jnp.mean(rho_cib, axis=-1, keepdims=True)
    viol_q   = jax.nn.relu(EPS_POS * q_cib - q_pred)     / q_cib   # local : relu(EPS_POS - q_pred / q_cib)
    return jnp.mean(jnp.sum(viol_rho + viol_q, axis=-1))


# u0s_batch, u_finals_batch : (batch, 3, n), vraies valeurs, u_final à T_bloc(u0) de u0
def _loss_stencil_conv1D(params, u0s_batch, u_finals_batch):
    Vs = jax.vmap(V_max)(u0s_batch)
    Rs = jax.vmap(R_max)(u0s_batch)
    u0s_batch      = jax.vmap(normalise)(u0s_batch, Vs, Rs)        # entrée et cible avec les mêmes V, R, ceux de l'entrée
    u_finals_batch = jax.vmap(normalise)(u_finals_batch, Vs, Rs)
    F_pred = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    u_pred = u0s_batch - LAMBDA * (F_pred - jnp.roll(F_pred, 1, axis=-1))

    erreur_vec = u_pred - u_finals_batch
    # norme L2 moyenne sur le batch, par variable : E ne domine pas rho, et rho*u peut être nul sur une trajectoire.
    # Sert aussi à loss_hf, quelle que soit PHYS_NORM (sinon l'échelle de loss_hf change d'un facteur ~sqrt(n))
    norme  = jnp.mean(jnp.sqrt(jnp.sum(u_finals_batch ** 2, axis=-1) + 1e-12), axis=0, keepdims=True)  # (1, 3)

    # L1 relative : régression vers la médiane, un choc décalé d'une cellule pèse moins -> moins d'étalement
    erreur   = jnp.sum(jnp.abs(erreur_vec), axis=-1)                                              # (batch, 3)
    norme_l1 = jnp.mean(jnp.sum(jnp.abs(u_finals_batch), axis=-1), axis=0, keepdims=True) + 1e-12  # (1, 3)
    loss_phys = jnp.mean(erreur / norme_l1)
   
    e_hat   = jnp.fft.rfft(erreur_vec, axis=-1)
    e_hf    = e_hat[..., k0:Ktot]
    loss_hf = 1 / (Ktot- k0) * jnp.mean(jnp.sum(jnp.abs(e_hf), axis=-1) / norme)

    loss_pos = loss_positivite(u_pred, u_finals_batch)

    loss = lambda_phys * loss_phys + lambda_hf * loss_hf + lambda_pos * loss_pos
    return loss, {"loss": loss, "loss_phys": loss_phys, "loss_hf": loss_hf, "loss_pos": loss_pos}

loss_fn = jax.jit(_loss_stencil_conv1D)

# diagnostic : termes BRUTS de la loss pour chaque paire, sans la division par la norme moyenne du batch
# (une norme par paire exploserait quand une composante est nulle, ex. u = 0 -> rho*u = 0 partout).
# La division par la norme moyenne sur tout le jeu se fait après (training.print_losses_par_type).
# (batch, 3, n) x2 -> dict de (batch, 3) : err_l1, norme_l1, hf, norme_l2 ; et loss_pos (batch,)
def _termes_par_ci(params, u0, uf):
    V  = V_max(u0)
    R  = R_max(u0)
    u0 = normalise(u0, V, R)
    uf = normalise(uf, V, R)
    F  = predict_F(params, u0)
    e  = u0 - LAMBDA * (F - jnp.roll(F, 1, axis=-1)) - uf
    return {"err_l1":   jnp.sum(jnp.abs(e), axis=-1),
            "norme_l1": jnp.sum(jnp.abs(uf), axis=-1),
            "hf":       jnp.sum(jnp.abs(jnp.fft.rfft(e, axis=-1)[..., k0:Ktot]), axis=-1) / (Ktot - k0),
            "norme_l2": jnp.sqrt(jnp.sum(uf ** 2, axis=-1) + 1e-12),
            "loss_pos": loss_positivite((uf + e)[None], uf[None])}

termes_par_ci = jax.jit(jax.vmap(_termes_par_ci, in_axes=(None, 0, 0)))

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, u_finals_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, u_finals_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step
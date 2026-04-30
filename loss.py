import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from network import FNO1D
from advection_solver import advection_solver, n_steps
from network_parameters import x, a

model = FNO1D(kmax=16, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64, di=1)

_, tfinal = advection_solver(jnp.ones(x.shape[0]), n_steps)

@jax.jit
def lossfn(F, F_cible):
    erreur = jnp.sqrt(jnp.sum((F - F_cible)**2, axis=1) + 1e-12)
    norme  = jnp.sqrt(jnp.sum(F_cible**2, axis=1) + 1e-12)
    return jnp.mean(erreur / norme)

def loss_fn(params, u0s_batch, F_batch, lambda_consistency):

    F_pred = jax.vmap(
        lambda u: model.apply(params, u))(u0s_batch)
    loss_main = lossfn(F_pred, F_batch)

    # Consistency loss
    # on definit des U0 entre -1 et 1 en excluant les valeurs proches de 0
    _half = u0s_batch.shape[0] // 2
    U0s = jnp.concatenate([
        jnp.linspace(-1., -0.1, _half),
        jnp.linspace( 0.1,  1., _half),
    ])

    # on crée des u0 constants à partir de ces U0s
    u0s_cst = jax.vmap(
        lambda U0: jnp.ones_like(u0s_batch[0]) * U0
    )(U0s)

    # on calcule les flux prédits pour ces u0 constants
    F_pred_cst = jax.vmap(lambda u: model.apply(params, u))(u0s_cst)

    # condition de consistance : F_pred_cst doit être égal à tfinal * a * U0 pour chaque U0
    F_cibles_cst = jax.vmap(
        lambda U0: jnp.ones_like(u0s_batch[0]) * (tfinal * a * U0)
    )(U0s)

    # ecrire le jnp.linalg.norm manuellement pour introduire un epsilon pour éviter les divisions par zéro 
    # (apres passafe au gradient)
    erreur_cst = jnp.sqrt(jnp.sum((F_pred_cst - F_cibles_cst)**2, axis=1) + 1e-12)
    norme_cst  = jnp.sqrt(jnp.sum(F_cibles_cst**2, axis=1) + 1e-12)
    loss_consistency = jnp.mean(erreur_cst / norme_cst)

    return loss_main + lambda_consistency * loss_consistency, {
    "loss_main": loss_main,
    "loss_consistency": lambda_consistency*loss_consistency
}

def make_train_step(optimizer, lambda_consistency):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, uts_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, uts_batch, lambda_consistency)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step

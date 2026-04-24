import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from network import FNO1D

model = FNO1D(kmax=16, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64, di=1)

@jax.jit
def lossfn(F, F_cible):
    erreur = jnp.linalg.norm(F - F_cible, axis=1)
    norme  = jnp.linalg.norm(F_cible, axis=1)
    return jnp.mean(erreur / norme)

@jax.jit
def loss_fn(params, u0s_batch, F_batch, lambda_consistency=10.0):

    F_pred = jax.vmap(
        lambda u: model.apply(params, u))(u0s_batch)
    loss_main = lossfn(F_pred, F_batch)

    # Consistency loss
    U0s = jnp.linspace(-1., 1., u0s_batch.shape[0])

    u0s_cst = jax.vmap(
        lambda U0: jnp.ones_like(u0s_batch[0]) * U0
    )(U0s)

    F_cst = jax.vmap(
        lambda u: model.apply(params, u)
    )(u0s_cst)
    D_cst = F_cst - jnp.roll(F_cst, 1, axis=1)
    loss_consistency = jnp.mean(D_cst ** 2)

    return loss_main + lambda_consistency*loss_consistency

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, uts_batch, lambda_consistency=10.0):
        grads = jax.grad(loss_fn)(params, u0s_batch, uts_batch, lambda_consistency=10.0)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step

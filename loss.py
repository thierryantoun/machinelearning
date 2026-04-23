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
def loss_fn(params, u0s_batch, F_batch):
    F_pred = jax.vmap(lambda u: model.apply(params, u))(u0s_batch)
    return lossfn(F_pred, F_batch)

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, uts_batch):
        grads = jax.grad(loss_fn)(params, u0s_batch, uts_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step

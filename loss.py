import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from network import FNO1D
from network_parameters import x, a, cfl, SOLVER

if SOLVER == "advection":
    from advection_solver import advection_solver as _solver, n_steps
else:
    from burgers_solver import burgers_solver as _solver, n_steps


def identity(x):
    return x


model = FNO1D(kmax=16, activation=nn.gelu, init_fn=nn.initializers.lecun_normal(), dv=64)

@jax.jit
def lossfn(F, F_cible):
    erreur = jnp.sqrt(jnp.sum((F - F_cible) ** 2, axis=1) + 1e-12)
    norme  = jnp.sqrt(jnp.sum(F_cible ** 2, axis=1) + 1e-12)
    return jnp.mean(erreur / norme)


def predict_F(params, u0):
    mean_u0 = jnp.mean(u0)
    N      = model.apply(params, u0)
    N_mean = model.apply(params, jnp.full_like(u0, mean_u0))
    if SOLVER == "advection":
        consistance = a * mean_u0
    else:
        consistance = mean_u0**2 / 2.0
    return consistance + N - N_mean


def loss_fn(params, u0s_batch, F_batch):
    F_total = jax.vmap(lambda u: predict_F(params, u))(u0s_batch)
    loss_main = lossfn(F_total, F_batch)
    return loss_main, {"loss_main": loss_main}


def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, u0s_batch, F_batch):
        grads, _ = jax.grad(loss_fn, has_aux=True)(params, u0s_batch, F_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state
    return train_step
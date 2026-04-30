import pickle
import jax.numpy as jnp
from jax import random

from network_parameters import K, x
from initial_data import generate_initial_data
from advection_solver import advection_solver, n_steps
from loss import model

with open("params.pkl", "rb") as f:
    params = pickle.load(f)

# test
u0 = generate_initial_data(random.PRNGKey(5), K)
F_cible, _ = advection_solver(u0, n_steps)
F_pred = model.apply(params, u0[:, None])
u_model = u0[:, None] - 1/(x[1] - x[0]) * (F_pred - jnp.roll(F_pred, 1, axis=0))
u_cible = u0 - 1/(x[1] - x[0]) * (F_cible - jnp.roll(F_cible, 1, axis=0))
# u_cible = advection_solver(u0_test)[0]

# MSE
mse = jnp.mean((u_model[:, 0] - u_cible) ** 2)
print(f"MSE entre u prédit et u cible : {mse:.6f}")

# fais un plot de u_test et u_cible pour comparer
import matplotlib.pyplot as plt
plt.plot(x, u_cible, label='u cible')
plt.plot(x, u_model, label='u prédit')
plt.plot(x,u0,label='u0')
plt.legend()
plt.title('Comparaison entre u cible et u prédit')
plt.xlabel('x')
plt.ylabel('u')
plt.show()
plt.savefig("comparaison_u.png")

# test de la consistance
U0 = 0.5
u0_cst = jnp.ones_like(x) * U0
F_cst = model.apply(params, u0_cst[:, None])
tfinal = advection_solver(jnp.ones_like(x), n_steps)[1]
F_cible_cst = jnp.ones_like(x) * (tfinal * U0)
D_cst = F_cst[:, 0] - F_cible_cst
consistency_loss = jnp.mean(D_cst)
print(f"Loss de consistance pour u0 constant : {consistency_loss:.6f}")
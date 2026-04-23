import pickle
import jax.numpy as jnp
from jax import random

from network_parameters import K, x
from initial_data import generate_initial_data
from advection_solver import advection_solver
from loss import model

with open("params.pkl", "rb") as f:
    params = pickle.load(f)

# test
u0_test = generate_initial_data(random.PRNGKey(123), K)
F_test_cible = advection_solver(u0_test)
F_test_pred = model.apply(params, u0_test[:, None])
u_test = u0_test[:, None] + (x[1] - x[0])*(F_test_pred - jnp.roll(F_test_pred, 1, axis=0))
u_cible = u0_test + (x[1] - x[0])*(F_test_cible - jnp.roll(F_test_cible, 1, axis=0))

# fais un plot de u_test et u_cible pour comparer
import matplotlib.pyplot as plt
plt.plot(x, u_cible, label='u cible')
plt.plot(x, u_test, label='u prédit')
plt.legend()
plt.title('Comparaison entre u cible et u prédit')
plt.xlabel('x')
plt.ylabel('u')
plt.show()

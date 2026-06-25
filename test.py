import pickle
import time
import jax
import jax.numpy as jnp
from jax import random

from network_parameters import K, x, n_steps, solver
from initial_data import generate_initial_data
from loss import model, predict_F

with open("params.pkl", "rb") as f:
    params = pickle.load(f)
    
import matplotlib.pyplot as plt

# test sur un créneau
# u0_creneau = jnp.where(x < 0.5, 0.0, 1.0)
u0_creneau = jnp.sin(2* jnp.pi * x)
u0_creneau_original = u0_creneau  # sauvegarde avant la boucle
multiple_steps = 1
#timer
t0 = time.perf_counter()
u_creneau, _, t_final = solver(u0_creneau, n_steps*multiple_steps)
jax.block_until_ready(u_creneau)
t1 = time.perf_counter()
print(f"Temps d'exécution du solveur d'advection pour {multiple_steps*n_steps} étapes : {t1 - t0:.4f} s")

dx = x[1] - x[0]

def step(u, _):
    F = predict_F(params, u)
    u_next = u - (t_final)/dx * (F - jnp.roll(F, 1, axis=0))
    return u_next, None

t2 = time.perf_counter()
u_pred_creneau, _ = jax.lax.scan(step, u0_creneau, None, length=multiple_steps)
jax.block_until_ready(u_pred_creneau)
t3 = time.perf_counter()
print(f"Temps d'exécution du modèle pour {multiple_steps*n_steps} étapes : {t3 - t2:.4f} s")

mse = jnp.mean((u_pred_creneau - u_creneau) ** 2)
print(f"[Créneau 0→1]                        MSE = {float(mse):.6f}")

plt.figure()
plt.plot(x, u0_creneau_original, label='u₀',    linestyle='--', alpha=0.5)
plt.plot(x, u_creneau, label='cible',  linewidth=1.5)
plt.plot(x, u_pred_creneau,  label='prédit', linewidth=1.5, linestyle=':')
plt.title("sin(2pix)")
plt.xlabel('x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_creneau_50.png", dpi=150)
plt.show()
print("Figure sauvegardée : test_creneau.png")

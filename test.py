import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from network_parameters import K, x, n_steps, cfl, SOLVER
from loss import predict_F

if SOLVER == "advection":
    from advection_solver import advection_solver as _active_solver
else:
    from burgers_solver import burgers_solver as _active_solver

solver = partial(_active_solver, n_steps=n_steps)

dx = x[1] - x[0]

with open("params.pkl", "rb") as f:
    params = pickle.load(f)

u0_creneau          = jnp.where(x < 0.5, 0.0, 1.0)
u0_creneau_original = u0_creneau
multiple_steps      = 100

# Solveur : tourne en blocs de n_steps pour récupérer le t réel de chaque bloc
t0 = time.perf_counter()
def solver_block(u, _):
    u_next, _, t = solver(u)
    return u_next, t
u_creneau, t_blocks = jax.lax.scan(solver_block, u0_creneau, None, length=multiple_steps)
jax.block_until_ready(u_creneau)
t1 = time.perf_counter()
print(f"Temps solveur pour {multiple_steps * n_steps} étapes : {t1 - t0:.4f} s")

# Modèle : utilise les mêmes t_blocks que le solveur
def step(u, t_block):
    F      = predict_F(params, u)
    u_next = u - t_block / dx * (F - jnp.roll(F, 1, axis=0))
    return u_next, None

t2 = time.perf_counter()
u_pred_creneau, _ = jax.lax.scan(step, u0_creneau, t_blocks)
jax.block_until_ready(u_pred_creneau)
t3 = time.perf_counter()
print(f"Temps modèle pour {multiple_steps * n_steps} étapes : {t3 - t2:.4f} s")

mse = jnp.mean((u_pred_creneau - u_creneau) ** 2)
print(f"[Créneau 0→1] MSE = {float(mse):.6f}")

plt.figure()
plt.plot(x, u0_creneau_original, label='u₀',    linestyle='--', alpha=0.5)
plt.plot(x, u_creneau,           label='cible',  linewidth=1.5)
plt.plot(x, u_pred_creneau,      label='prédit', linewidth=1.5, linestyle=':')
plt.title("Créneau")
plt.xlabel('x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_creneau_final.png", dpi=150)
plt.show()
print("Figure sauvegardée : test_creneau_final.png")

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from euler_solver import euler_solver, primitives, gamma

# Blast wave (type Woodward-Colella) sur domaine periodique [0, 1)
# rho = 1, u = 0 partout ; p = 1000 a gauche (x<0.1), p = 100 a droite (x>0.9), p = 0.01 au milieu
# (en periodique, les deux zones de haute pression sont voisines a travers le bord)
nx = 800
T = 0.012
x = (jnp.arange(nx) + 0.5) / nx

rho = jnp.ones(nx)
u = jnp.zeros(nx)
p = jnp.where(x < 0.1, 1000.0, jnp.where(x > 0.9, 100.0, 0.01))
U0 = jnp.stack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2])

U, F_moy, _ = euler_solver(U0, T)
rho_f, u_f, p_f = primitives(U)

print("finite :", bool(jnp.all(jnp.isfinite(U))))
print("rho min/max :", float(rho_f.min()), float(rho_f.max()))
print("p   min/max :", float(p_f.min()), float(p_f.max()))
print("u   min/max :", float(u_f.min()), float(u_f.max()))
# conservation (domaine periodique => masse, qdm, energie conservees)
for name, i in [("masse", 0), ("qdm", 1), ("energie", 2)]:
    print(f"{name:8s} : {float(U0[i].sum()/nx):.12e} -> {float(U[i].sum()/nx):.12e}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, y0, y, lab in zip(ax, [rho, u, p], [rho_f, u_f, p_f], ["rho", "u", "p"]):
    a.plot(x, y0, "k--", lw=0.8, label="t=0")
    a.plot(x, y, "b-", lw=1.0, label=f"t={T}")
    a.set_title(lab); a.set_xlabel("x"); a.legend()
ax[2].set_yscale("log")
plt.tight_layout()
plt.savefig("blast_test.png", dpi=120)
print("figure -> blast_test.png")

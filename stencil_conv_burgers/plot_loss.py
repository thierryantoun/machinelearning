"""Trace les losses train / val écrites par training.py dans losses.txt (une ligne toutes les 10 epochs).
Utilisable pendant le training. Usage : python plot_loss.py [losses.txt]"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "losses.txt"
with open(path) as f:
    header = f.readline().strip().split("\t")
data = np.atleast_2d(np.loadtxt(path, skiprows=1))
cols = {name: data[:, i] for i, name in enumerate(header)}
epochs = cols["epoch"]
metrics = [k[len("train_"):] for k in header if k.startswith("train_")]   # loss, loss_phys, loss_hf, ...

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"

fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
for ax, m in zip(axes[0], metrics):
    for split, color, ls in (("train", TRAIN_COLOR, "-"), ("val", VAL_COLOR, "--")):
        y = cols[f"{split}_{m}"]
        ax.plot(epochs, y, color=color, lw=2, ls=ls, label=split.capitalize())
        ax.plot(epochs[-1], y[-1], "o", ms=5, color=color, mec="white", mew=1.2)
        ax.annotate(f"{y[-1]:.4g}", (epochs[-1], y[-1]), xytext=(-6, 6 if split == "val" else -6),
                    textcoords="offset points", ha="right", va="bottom" if split == "val" else "top",
                    color=INK, fontsize=8)
    if np.all(cols[f"train_{m}"] > 0) and np.all(cols[f"val_{m}"] > 0):
        ax.set_yscale("log")
    ax.set_title(m, color=INK, loc="left")
    ax.set_xlabel("Epoch", color=MUTED)
    ax.grid(True, which="major", color="#e6e5e1", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
axes[0][0].legend(frameon=False)

fig.tight_layout()
out = path.rsplit(".", 1)[0] + ".png"
fig.savefig(out, dpi=150)
print(f"figure -> {out}")

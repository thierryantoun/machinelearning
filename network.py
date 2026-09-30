import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from network_parameters import MODEL, K


# ---------------------------------------------------------------------------
# Outils spectraux (norm="forward" : rfft donne les vrais coefficients c_k,
# irfft évalue la série sur n points sans facteur d'échelle)
# ---------------------------------------------------------------------------

def to_modes(x, kmax):
    """Grille (n, C) -> coefficients c_0..c_{kmax-1}, forme (kmax, C)."""
    return jnp.fft.rfft(x, axis=0, norm="forward")[:kmax]


def from_modes(c, n):
    """Coefficients (kmax, C) -> valeurs sur n points, forme (n, C).
    irfft complète les modes manquants par des zéros."""
    return jnp.fft.irfft(c, n=n, axis=0, norm="forward")


def aa_activation(z, activation, kmax, M):
    """Activation anti-aliasée.
    z : (N, C), supposé dans la bande |k| < kmax.
    1. coefficients exacts de z ;
    2. évaluation sur une grille fine fixe de M points ;
    3. activation point par point sur cette grille ;
    4. retour dans la bande |k| < kmax ;
    5. retour sur la grille de stockage N."""
    N = z.shape[0]
    c = to_modes(z, kmax)                 # (kmax, C)
    z_fine = from_modes(c, M)             # (M, C)
    s_fine = activation(z_fine)           # (M, C)
    c_out = to_modes(s_fine, kmax)        # (kmax, C)
    return from_modes(c_out, N)           # (N, C)


def default_M(kmax):
    """Grille fine : au moins 8 * kmax, arrondie à la puissance de 2 suivante.
    Dépend de kmax, jamais de N."""
    return 2 ** math.ceil(math.log2(8 * kmax))


@jax.jit
def multiply_one_mode(R_k, v_k):
    return R_k @ v_k


class FNOBlock(nn.Module):
    kmax: int
    width: int
    activation: callable
    init_fn: callable
    M: int

    def setup(self):
        self.W = nn.Dense(features=self.width, use_bias=False)
        self.R_real = self.param('R_real', self.init_fn, (self.kmax, self.width, self.width))
        self.R_imag = self.param('R_imag', self.init_fn, (self.kmax, self.width, self.width))

    def __call__(self, v):                                   # v : (N, width), dans la bande
        R = self.R_real + 1j * self.R_imag
        Wv = self.W(v)                                       # ponctuel : reste dans la bande
        v_hat = jnp.fft.rfft(v, axis=0)[:self.kmax, :]
        RFv = jax.vmap(multiply_one_mode)(R, v_hat)
        RFv_full = jnp.zeros((v.shape[0] // 2 + 1, self.width), dtype=jnp.complex64)
        RFv_full = RFv_full.at[:self.kmax, :].set(RFv)
        Finverse = jnp.fft.irfft(RFv_full, n=v.shape[0], axis=0)
        # --- changement : activation anti-aliasée au lieu de self.activation(...)
        return aa_activation(Wv + Finverse, self.activation, self.kmax, self.M)


class FNO1D(nn.Module):
    kmax: int
    activation: callable
    init_fn: callable
    dv: int = 64
    M: int = 0              # 0 -> default_M(kmax)

    def setup(self):
        M = self.M if self.M > 0 else default_M(self.kmax)
        assert M >= 2 * self.kmax, "la grille fine doit contenir la bande"
        self.lifting = nn.Dense(features=self.dv, use_bias=True)
        self.block1 = FNOBlock(kmax=self.kmax, width=self.dv,
                               activation=self.activation,
                               init_fn=self.init_fn, M=M)
        self.block2 = FNOBlock(kmax=self.kmax, width=self.dv,
                               activation=self.activation,
                               init_fn=self.init_fn, M=M)
        self.block3 = FNOBlock(kmax=self.kmax, width=self.dv,
                               activation=self.activation,
                               init_fn=self.init_fn, M=M)
        self.block4 = FNOBlock(kmax=self.kmax, width=self.dv,
                               activation=self.activation,
                               init_fn=self.init_fn, M=M)
        self.blocks = [self.block1, self.block2, self.block3, self.block4]
        self.projection = nn.Dense(features=1, use_bias=True)

    def __call__(self, u):
        N = u.shape[0]
        assert N >= 2 * self.kmax, "stockage exact : il faut N >= 2 kmax"
        x = u[:, None]
        x = from_modes(to_modes(x, self.kmax), N)   # filtre d'entrée, même bande
        x = self.lifting(x)                         # affine ponctuel : reste dans la bande
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)                      # affine ponctuel : reste dans la bande
        return x[:, 0]


model = FNO1D(kmax=K, activation=nn.gelu,
              init_fn=nn.initializers.lecun_normal(), dv=64)